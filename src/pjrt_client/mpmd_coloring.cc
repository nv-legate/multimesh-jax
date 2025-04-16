/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_coloring.h"

#include <optional>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/json_utils.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"
#include "xla/service/call_inliner.h"
#include "xla/service/tuple_simplifier.h"

namespace xla {
namespace {

struct TaskConfig {
  std::string name;
  std::vector<int64_t> devices;
  std::optional<LogicalShardingContext> autosharding;
};

template <typename K, typename V>
V value_or(const absl::flat_hash_map<K, V>& m, const K& k, V v) {
  auto iter = m.find(k);
  if (iter == m.end()) {
    return v;
  }
  return iter->second;
}

// Returns a task config for the `json` node.
// `context` gives a debug description for errors.
absl::StatusOr<TaskConfig> GetTaskConfig(const Json::Value& json,
                                         const std::string& context) {
  TF_ASSIGN_OR_RETURN(auto name,
                      GetTaskValue<std::string>(json, context, "name"));
  TF_ASSIGN_OR_RETURN(auto devices, GetTaskValue<std::vector<int64_t>>(
                                        json, context, "devices"));
  TF_ASSIGN_OR_RETURN(
      std::optional<int64_t> loop_submesh_size,
      GetOptionalTaskValue<int64_t>(json, context, "loop_submesh_size"));
  TF_ASSIGN_OR_RETURN(
      bool loop_submesh_reverse,
      GetOptionalTaskValue(json, context, "loop_submesh_reverse", false));

  auto autosharding_json = json.get("autosharding", Json::Value::null);
  std::optional<LogicalShardingContext> autosharding;
  if (!autosharding_json.isNull()) {
    TF_ASSIGN_OR_RETURN(autosharding, GetLogicalShardingContext(
                                          autosharding_json, context, devices));
  }

  if (loop_submesh_size.has_value()) {
    autosharding->loop_submesh =
        LoopDependentSubmesh({.task_mesh_size = *loop_submesh_size,
                              .global_mesh_start = devices.front(),
                              .global_mesh_stop = devices.back() + 1,
                              .reverse = loop_submesh_reverse});
  }

  return TaskConfig{
      .name = std::move(name),
      .devices = std::move(devices),
      .autosharding = std::move(autosharding),
  };
}

// Computes a weight for a given instruction based on the byte size
// of the instruction. Operations like broadcast have lower weight
// since they can be reconstructed from a smaller operand
// inside a fusion. This occurs before sharding propagation,
// which means assuming the same sharding amount.
int64_t OperandWeight(const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kBroadcast) {
    return OperandWeight(instruction->operand(0));
  }
  return ShapeUtil::ElementsIn(instruction->shape());
}

bool AllowOverride(int64_t index, absl::Span<const bool> override) {
  if (override.size() > index) {
    return override[index];
  }
  if (override.empty()) {
    return false;
  }
  return override[0];
}

absl::Status HandleRootTupleShardings(HloPartition* partition,
                                      HloModule* module, HloInstruction* root) {
  if (!root->has_sharding() || !root->shape().IsTuple()) {
    return absl::OkStatus();
  }

  for (int64_t index = 0; index < root->operand_count(); ++index) {
    const HloSharding& sharding = root->sharding().tuple_elements()[index];
    auto* operand = root->mutable_operand(index);
    const bool allow_sharding_overwrite = AllowOverride(
        index, module->config().allow_spmd_sharding_propagation_to_output());
    if (!sharding.IsReplicated() && !operand->has_sharding()) {
      operand->set_sharding(sharding);
    }

    if (!allow_sharding_overwrite) {
      VLOG(5) << operand->name() << " is root operand " << index
              << ", which must use fixed sharding " << sharding;
      // we have to create a coloring here to make sure that the correct output
      // sharding is used for this
      HloInstruction* recolor =
          root->parent()->AddInstruction(HloInstruction::CreateCustomCall(
              operand->shape(), {operand}, kCustomCallRootTupleRecolor));
      TF_RETURN_IF_ERROR(root->ReplaceOperandWith(index, recolor));
      std::string color = [&] {
        if (sharding.IsReplicated()) {
          return *partition->FindOrAllocateGlobalColor();
        }
        zuku::DeviceList devices =
            *CreateDeviceList(sharding.tile_assignment());
        return *partition->FindOrAllocateColor(devices, nullptr);
      }();
      recolor->set_sharding(sharding);
      AssignColor(recolor, color);
    }
  }
  return absl::OkStatus();
}

// Given a partition assigment `color` for the given `instruction`,
// propagate colorings backward to aliases in the
// `properties` map.
void ColorBackwards(const std::string& color, HloInstruction* instruction,
                    const InstructionProperties& properties) {
  if (!IsAssignedColor(instruction)) {
    LOG(FATAL) << instruction->name()
               << " was not assigned color before coloring backward";
  }

  std::vector<HloInstruction*> to_visit;
  auto add_visits = [&](HloInstruction* next) {
    for (auto* operand : next->mutable_operands()) {
      // don't propagate back to argument tuples
      if (operand->opcode() != HloOpcode::kParameter ||
          !operand->shape().IsTuple()) {
        to_visit.push_back(operand);
      }
    }
    properties.ForEachBackwardAlias(
        next, [&](HloInstruction* i) { to_visit.push_back(i); });
  };

  add_visits(instruction);
  while (!to_visit.empty()) {
    HloInstruction* next = to_visit.back();
    to_visit.pop_back();
    if (instruction->IsCustomCall(kCustomCallSliceOffset)) {
      continue;
    }

    if (properties.DerivedConstant(next)) {
      continue;
    }

    if (!IsAssignedColor(next)) {
      VLOG(5) << "coloring backwards to " << next->name() << ",color=" << color
              << " from root " << instruction->name();
      AssignColor(next, color);
      add_visits(next);
    }
  }
}

}  // namespace

absl::Status MpmdColoring::ColorTuple(HloInstruction* instruction) {
  std::optional<std::string> uniform_color{std::nullopt};
  for (auto* operand : instruction->operands()) {
    auto operand_color = Color(operand);
    if (operand_color.has_value()) {
      if (!uniform_color.has_value()) {
        uniform_color = Color(operand);
      } else if (*uniform_color != *operand_color) {
        uniform_color = std::nullopt;
        break;
      }
    } else {
      uniform_color = std::nullopt;
      break;
    }
  }

  // tuples should only be assigned a uniform color when ALL of the
  // operands have been assigned a color and that color is the same
  if (uniform_color.has_value()) {
    VLOG(5) << "assigning uniform color " << *uniform_color << " to tuple "
            << instruction->name();
    AssignColor(instruction, *std::move(uniform_color));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::optional<std::string>>
MpmdColoring::CheckForExplicitShardingColor(
    HloInstruction* instruction, const InstructionProperties& properties) {
  auto explicit_sharding = [&]() -> std::optional<HloSharding> {
    if (instruction->has_sharding()) {
      return instruction->sharding();
    }
    return std::nullopt;
  }();

  if (!explicit_sharding.has_value()) {
    return std::nullopt;
  }

  if (explicit_sharding->IsReplicated()) {
    // make sure this is not a parameter that has been explicitly assigned
    // replicated sharding
    std::optional<int64_t> parameter_number =
        properties.ParameterNumber(instruction);
    if (parameter_number.has_value()) {
      if (AllowOverride(*parameter_number,
                        instruction->parent()
                            ->parent()
                            ->config()
                            .allow_spmd_sharding_propagation_to_parameters())) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }

  zuku::DeviceList devices = [&] {
    if (explicit_sharding->IsReplicated()) {
      return partition_->Devices();
    }
    return *CreateDeviceList(explicit_sharding->tile_assignment());
  }();

  auto devices_color = partition_->FindColor(devices);
  if (!devices_color.has_value()) {
    TF_ASSIGN_OR_RETURN(devices_color, partition_->AllocateColor(
                                           "sharding", std::move(devices)));
  }
  VLOG(5) << instruction->name() << " assigned from sharding "
          << *explicit_sharding << " to color=" << *devices_color;

  return std::move(devices_color);
}

absl::StatusOr<bool> MpmdColoring::InlineExplicitTasks(
    HloComputation* computation) {
  bool changed = false;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->IsCustomCall("LegateTask")) {
      VLOG(5) << "coloring explicit task instruction " << instruction->name();

      changed = true;

      auto json = GetJsonValue(instruction->raw_backend_config_string().data(),
                               instruction->raw_backend_config_string().size());
      if (!json.ok()) {
        return InvalidArgumentStrCat(instruction->name(),
                                     " has bad json config");
      }

      TF_ASSIGN_OR_RETURN(
          TaskConfig config,
          GetTaskConfig(*json, std::string(instruction->name())));

      auto dl = [&]() -> absl::StatusOr<zuku::DeviceList> {
        if (config.devices.empty()) {
          return partition_->DefaultDevices();
        }
        return CreateDeviceList(config.devices);
      }();

      std::shared_ptr<LogicalShardingContext> context;
      if (config.autosharding.has_value()) {
        context = std::make_shared<LogicalShardingContext>(
            *std::move(config.autosharding));
      }

      TF_ASSIGN_OR_RETURN(const std::string color,
                          partition_->AllocateColor(config.name, *std::move(dl),
                                                    std::move(context)));

      for (auto* sub : instruction->called_computations()[0]->instructions()) {
        VLOG(5) << sub->name() << " assigned color=" << color
                << " from task custom-call";
        AssignColor(sub, color);
      }
      // turn the task into a regular call and inline
      auto* call_to_inline =
          computation->AddInstruction(HloInstruction::CreateCall(
              instruction->shape(), instruction->operands(),
              instruction->called_computations()[0]));

      TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(call_to_inline));

      AssignColor(instruction, color);
    } else {
      for (auto* comp : instruction->called_computations()) {
        TF_ASSIGN_OR_RETURN(bool comp_changed, InlineExplicitTasks(comp));
        changed |= comp_changed;
      }
    }
  }
  return changed;
}

absl::Status MpmdColoring::ComputeAssignedColors(
    HloComputation* computation, const InstructionProperties& properties) {
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction == computation->root_instruction() &&
        instruction->opcode() == HloOpcode::kTuple) {
      continue;
    }

    auto color = Color(instruction);
    if (color.has_value()) {
      // make sure the color is allocated in the HLO partition
      if (!partition_->HasColor(*color)) {
        TF_RETURN_IF_ERROR(partition_->AllocateColor(*color));
      }
    }

    if (!color.has_value() && instruction->opcode() == HloOpcode::kTuple) {
      TF_RETURN_IF_ERROR(ColorTuple(instruction));
      continue;
    }

    if (!color.has_value()) {
      TF_ASSIGN_OR_RETURN(
          color, CheckForExplicitShardingColor(instruction, properties));
    }

    if (!color.has_value()) {
      TF_ASSIGN_OR_RETURN(color,
                          partition_->ComputeMetadataNameColor(instruction));
      if (color.has_value()) {
        VLOG(5) << "computed metadata name color for " << instruction->name()
                << ",color=" << *color
                << ",metadata=" << instruction->metadata().op_name();
      }
    }

    if (color.has_value()) {
      AssignColor(instruction, *color);
      properties.ForEachAlias(instruction, [&](HloInstruction* i) {
        if (!IsAssignedColor(i)) {
          AssignColor(i, *color);
        }
      });
    }

    if (instruction->opcode() == HloOpcode::kWhile) {
      TF_RETURN_IF_ERROR(ComputeAssignedColors(
          instruction->called_computations()[0], properties));
    }
  }
  return absl::OkStatus();
}

bool MpmdColoring::PropagateFromUsersAndOperands(
    HloInstruction* instruction, const InstructionProperties& properties,
    FilterVisitFn if_visit,
    const absl::flat_hash_map<const HloInstruction*, int64_t>& depth,
    const absl::flat_hash_map<const HloInstruction*, int64_t>&
        topological_index) {
  absl::flat_hash_map<std::string, int64_t>
      color_weights;  // reuse to avoid re-allocating memory
  absl::flat_hash_map<std::string, int64_t>
      color_depths;  // reuse to avoid re-allocating memory
  absl::flat_hash_map<std::string, int64_t>
      color_topological_indices;  // reuse to avoid re-allocating memory
  auto color = Color(instruction);
  std::optional<std::string> max_operand_color;
  // Max is a misnomer here for depth and topological index,
  // it actually refers to the fact that the color has the highest
  // score.
  int64_t max_color_weight = 0;
  int64_t max_color_depth = std::numeric_limits<int64_t>::max();
  int64_t max_color_topological_index = std::numeric_limits<int64_t>::max();

  // Needed to get value_or to work with const HloInstruction*
  const HloInstruction* const_instruction = instruction;
  const int64_t instruction_depth =
      value_or(depth, const_instruction, std::numeric_limits<int64_t>::max());
  const int64_t instruction_topo_index =
      value_or(topological_index, const_instruction,
               std::numeric_limits<int64_t>::max());

  auto add_user_or_operand = [&](const HloInstruction* i) {
    if (if_visit(i)) {
      if (i->opcode() == HloOpcode::kBroadcast) {
        // reweight since broadcasts make operands trivially bigger
        i = i->operand(0);
      }

      auto color = Color(i);
      if (color.has_value()) {
        const int64_t weight =
            ShapeUtil::ByteSizeOf(i->shape(), /*pointer_size=*/sizeof(void*));
        const int64_t i_depth =
            value_or(depth, i, std::numeric_limits<int64_t>::max());
        const int64_t i_topo_index =
            value_or(topological_index, i, std::numeric_limits<int64_t>::max());
        VLOG(5) << instruction->name() << " has operand " << i->name()
                << " with color=" << *color << " has weight=" << weight
                << ",depth=" << i_depth << ",topo_index=" << i_topo_index;
        color_weights[*color] =
            value_or(color_weights, *color, int64_t(0)) + weight;
        color_depths[*color] = std::min(
            value_or(color_depths, *color, std::numeric_limits<int64_t>::max()),
            std::abs(i_depth - instruction_depth));
        color_topological_indices[*color] =
            std::min(value_or(color_topological_indices, *color,
                              std::numeric_limits<int64_t>::max()),
                     std::abs(i_topo_index - instruction_topo_index));
        bool update_max_color = false;
        switch (color_propagation_priority_) {
          case ColorPropagationPriority::kWeight:
            // Check weight, then depth, then topological index
            update_max_color = !max_operand_color.has_value() ||
                               color_weights[*color] > max_color_weight ||
                               (color_weights[*color] == max_color_weight &&
                                (color_depths[*color] < max_color_depth ||
                                 (color_depths[*color] == max_color_depth &&
                                  color_topological_indices[*color] <
                                      max_color_topological_index)));
            break;
          case ColorPropagationPriority::kDepth:
            // Check depth, then weight, then topological index
            update_max_color = !max_operand_color.has_value() ||
                               color_depths[*color] < max_color_depth ||
                               (color_depths[*color] == max_color_depth &&
                                (color_weights[*color] > max_color_weight ||
                                 (color_weights[*color] == max_color_weight &&
                                  color_topological_indices[*color] <
                                      max_color_topological_index)));
            break;
          case ColorPropagationPriority::kTopological:
            // Check topological index ONLY since it is well ordered.
            update_max_color =
                !max_operand_color.has_value() ||
                color_topological_indices[*color] < max_color_topological_index;
            break;
          default:
            throw std::invalid_argument(
                "Unknown color propagation priority! It should never be "
                "possible to reach this point.");
        }
        if (update_max_color) {
          max_operand_color = color;
          max_color_weight = color_weights[*color];
          max_color_depth = color_depths[*color];
          max_color_topological_index = color_topological_indices[*color];
        }
      }
    }
  };

  for (auto* operand : properties.Operands(instruction)) {
    add_user_or_operand(operand);
  }
  if (instruction->opcode() != HloOpcode::kOptimizationBarrier) {
    for (auto* user : properties.Users(instruction)) {
      add_user_or_operand(user);
    }
  }

  if (max_operand_color.has_value()) {
    VLOG(5) << instruction->name() << " " << instruction->shape()
            << " assigning itself color " << *max_operand_color
            << " based on operand/user coloring";
    AssignColor(instruction, *max_operand_color);
    properties.ForEachAlias(instruction, [&](HloInstruction* i) {
      if (!IsAssignedColor(i)) {
        AssignColor(i, *max_operand_color);
      }
    });
    return true;
  }
  return false;
}

MpmdColoring::ColorPropagationPriority
MpmdColoring::GetColorPropagationPriorityFromString(
    absl::string_view priority_str) {
  if (priority_str == "WEIGHT") {
    VLOG(5) << "using weight priority for color propagation.";
    return MpmdColoring::ColorPropagationPriority::kWeight;
  } else if (priority_str == "DEPTH") {
    VLOG(5) << "using depth priority for color propagation.";
    return MpmdColoring::ColorPropagationPriority::kDepth;
  } else if (priority_str == "TOPOLOGICAL") {
    VLOG(5) << "using topological priority for color propagation.";
    return MpmdColoring::ColorPropagationPriority::kTopological;
  } else {
    VLOG(5) << "invalid priority string, using weight priority for color "
               "propagation.";
    return MpmdColoring::ColorPropagationPriority::kWeight;
  }
}

static absl::flat_hash_map<const HloInstruction*, int64_t> ComputeDepthMap(
    const HloComputation* computation) {
  absl::flat_hash_map<const HloInstruction*, int64_t> depth;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    depth[instruction] = 0;
    for (auto* operand : instruction->operands()) {
      depth[instruction] = std::max(depth[instruction], depth[operand] + 1);
    }
  }
  return depth;
}

static absl::flat_hash_map<const HloInstruction*, int64_t>
CreateTopologicalIndexMap(const HloComputation* computation) {
  absl::flat_hash_map<const HloInstruction*, int64_t> instruction_to_topo_index;
  const std::vector<HloInstruction*> topological_order =
      computation->MakeInstructionPostOrder();
  for (int64_t i = 0; i < topological_order.size(); ++i) {
    instruction_to_topo_index[topological_order[i]] = i;
  }
  return instruction_to_topo_index;
}

absl::StatusOr<bool> MpmdColoring::PropagateIf(
    HloComputation* computation, const InstructionProperties& properties,
    FilterVisitFn if_visit, bool microbatch_loop) {
  auto postorder = computation->MakeInstructionPostOrder();
  auto depth = ComputeDepthMap(computation);
  auto topological_index = CreateTopologicalIndexMap(computation);

  bool propagated = true;
  bool changed = false;
  while (propagated) {
    propagated = false;
    for (auto* instruction : postorder) {
      // these take special care to only assign themselves colors when all of
      // their operands are uniformly colored
      if (instruction->opcode() == HloOpcode::kTuple ||
          instruction->opcode() == HloOpcode::kOptimizationBarrier) {
        TF_RETURN_IF_ERROR(ColorTuple(instruction));
        continue;
      }

      if (if_visit(instruction) && !IsAssignedColor(instruction)) {
        VLOG(5) << "visiting " << instruction->name()
                << " in color propagation from operands/users";
        propagated |= PropagateFromUsersAndOperands(
            instruction, properties, if_visit, depth, topological_index);
      }
      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_ASSIGN_OR_RETURN(
            bool comp_propagated,
            PropagateIf(instruction->called_computations()[0], properties,
                        if_visit, /*microbatch_loop=*/microbatch_loop ||
                                      instruction->has_backend_config()));
        propagated |= comp_propagated;
      }
    }
    changed |= propagated;
  }

  return changed;
}

absl::StatusOr<bool> MpmdColoring::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto* root = module->entry_computation()->root_instruction();
  // we use sharding in the coloring so we do a (tiny) bit of sharding
  // propagation if a root tuple has explicit sharding annotations, propagate
  // them to the tuple operands
  TF_RETURN_IF_ERROR(HandleRootTupleShardings(partition_, module, root));

  TF_ASSIGN_OR_RETURN(bool changed,
                      InlineExplicitTasks(module->entry_computation()));
  if (changed) {
    CallInliner inliner;
    TF_ASSIGN_OR_RETURN(bool _, inliner.Run(module));

    TupleSimplifier simplifier;
    TF_ASSIGN_OR_RETURN(_, simplifier.Run(module));
  }

  auto properties = InstructionProperties::Create(module);

  TF_RETURN_IF_ERROR(
      ComputeAssignedColors(module->entry_computation(), properties));

  // first only visit elementwise propagation
  TF_ASSIGN_OR_RETURN(bool propagated_elementwise,
                      PropagateIf(
                          module->entry_computation(), properties,
                          [](const HloInstruction* i) {
                            return i->opcode() != HloOpcode::kConstant &&
                                   !i->shape().IsTuple() &&
                                   (i->IsElementwise() ||
                                    i->opcode() == HloOpcode::kParameter ||
                                    i->opcode() == HloOpcode::kReduce);
                          },
                          /*microbatch_loop=*/false));

  // next only visit non-trivial instructions
  TF_ASSIGN_OR_RETURN(bool propagated_nontrivial_shapes,
                      PropagateIf(
                          module->entry_computation(), properties,
                          [](const HloInstruction* i) {
                            return i->opcode() != HloOpcode::kConstant &&
                                   !i->shape().IsTuple() &&
                                   ShapeUtil::ElementsIn(i->shape()) > 1;
                          },
                          /*microbatch_loop=*/false));

  // now visit everything
  TF_ASSIGN_OR_RETURN(
      bool propagated_any,
      PropagateIf(
          module->entry_computation(), properties,
          [](const HloInstruction* i) {
            return i->opcode() != HloOpcode::kConstant &&
                   (i->opcode() == HloOpcode::kRngBitGenerator ||
                    !i->shape().IsTuple());
          },
          /*microbatch_loop=*/false));

  auto color_root = [&](HloInstruction* i) {
    if (!IsAssignedColor(i)) {
      // a root must always be assigned a color, if all else has failed
      // give a default color
      TF_ASSIGN_OR_RETURN(std::string color,
                          partition_->FindOrAllocateDefaultColor());
      VLOG(5) << "root " << i->name() << " assigned default color=" << color;
      AssignColor(i, color);
      ColorBackwards(color, i, properties);
    }
    return absl::OkStatus();
  };
  if (root->shape().IsTuple()) {
    for (auto* operand : root->mutable_operands()) {
      TF_RETURN_IF_ERROR(color_root(operand));
    }
  } else {
    TF_RETURN_IF_ERROR(color_root(root));
  }

  // for now, assume coloring always changes the module
  return true;
}

}  // namespace xla
