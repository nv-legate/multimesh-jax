/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_coloring.h"

#include <optional>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

#include "xla/pjrt/multimesh/mpmd_inline_explicit_tasks.h"
#include "xla/pjrt/multimesh/mpmd_insert_root_tuple_shardings.h"
#include "xla/pjrt/multimesh/mpmd_compute_assigned_colors.h"
#include "xla/pjrt/multimesh/mpmd_unpack_optimization_barrier.h"
#include "xla/pjrt/multimesh/mpmd_repack_optimization_barrier.h"

namespace xla {
namespace {

template <typename K, typename V>
V value_or(const absl::flat_hash_map<K, V>& m, const K& k, V v) {
  auto iter = m.find(k);
  if (iter == m.end()) {
    return v;
  }
  return iter->second;
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

bool MpmdColoring::PropagateFromUsersAndOperandsColorDepth(
    HloInstruction* instruction, const InstructionProperties& properties,
    FilterVisitFn if_visit,
    const absl::flat_hash_map<std::string, int64_t>& color_depth,
    absl::flat_hash_map<HloInstruction*, bool>& recolorable_instructions) {
  auto color = Color(instruction);

  // Check if non-recolorable instruction is already assigned a color
  if (color.has_value() && !recolorable_instructions.contains(instruction)) {
    return false;
  }

  std::optional<std::string> operand_candidate_color = std::nullopt;
  std::optional<std::string> user_candidate_color = std::nullopt;

  for (auto* operand : properties.Operands(instruction)) {
    const auto operand_color = Color(operand);
    if (if_visit(operand) && operand_color.has_value() &&
        color_depth.contains(*operand_color)) {
      if (!operand_candidate_color.has_value() ||
          color_depth.at(*operand_candidate_color) <
              color_depth.at(*operand_color)) {
        operand_candidate_color = operand_color;
      }
    }
  }

  for (auto* user : properties.Users(instruction)) {
    const auto user_color = Color(user);
    if (if_visit(user) && user_color.has_value() &&
        color_depth.contains(*user_color)) {
      if (!user_candidate_color.has_value() ||
          color_depth.at(*user_candidate_color) > color_depth.at(*user_color)) {
        user_candidate_color = user_color;
      }
    }
  }

  bool prefer_users_over_operands = false;
  if (recolorable_instructions.contains(instruction)) {
    prefer_users_over_operands = recolorable_instructions.at(instruction);
  } else if (user_candidate_color.has_value()) {
    prefer_users_over_operands = true;
  }

  std::optional<std::string> candidate_color = prefer_users_over_operands
                                                   ? user_candidate_color
                                                   : operand_candidate_color;
  // Check if the candidate color is deeper than the user candidate color. Not
  // ok to use in that case.
  if (candidate_color.has_value() && user_candidate_color.has_value()) {
    if (color_depth.at(*candidate_color) >
        color_depth.at(*user_candidate_color)) {
      candidate_color = user_candidate_color;
    }
  }

  if (candidate_color.has_value()) {
    VLOG(5) << "propagating color " << *candidate_color << " for "
            << instruction->name();
    AssignColor(instruction, *candidate_color);
    properties.ForEachAlias(instruction, [&](HloInstruction* i) {
      if (!IsAssignedColor(i)) {
        AssignColor(i, *candidate_color);
      }
    });
    recolorable_instructions[instruction] = prefer_users_over_operands;
    return candidate_color != color;
  }

  return false;
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
  for (auto* user : properties.Users(instruction)) {
    add_user_or_operand(user);
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
  } else if (priority_str == "COLOR_DEPTH") {
    VLOG(5) << "using color depth priority for color propagation.";
    return MpmdColoring::ColorPropagationPriority::kColorDepth;
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

static absl::flat_hash_map<std::string, int64_t> ComputeColorDepthMap(
    const HloComputation* computation,
    const absl::flat_hash_map<const HloInstruction*, int64_t>& depth) {
  absl::flat_hash_map<std::string, int64_t> color_depth;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    auto inst_color = Color(instruction);
    if (inst_color.has_value()) {
      color_depth[*inst_color] =
          std::max(value_or(color_depth, *inst_color, int64_t(0)),
                   depth.at(instruction));
    }
  }
  return color_depth;
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
  auto color_depth = ComputeColorDepthMap(computation, depth);

  absl::flat_hash_map<HloInstruction*, bool> recolorable_instructions;

  bool propagated = true;
  bool changed = false;
  while (propagated) {
    propagated = false;
    for (auto* instruction : postorder) {
      // these take special care to only assign themselves colors when all of
      // their operands are uniformly colored
      if (instruction->opcode() == HloOpcode::kTuple) {
        ColorTuple(instruction);
        continue;
      }

      if (if_visit(instruction) &&
          (!IsAssignedColor(instruction) ||
           recolorable_instructions.contains(instruction))) {
        VLOG(5) << "visiting " << instruction->name()
                << " in color propagation from operands/users";
        propagated |= [&]() {
          if (color_propagation_priority_ ==
              ColorPropagationPriority::kColorDepth) {
            return PropagateFromUsersAndOperandsColorDepth(
                instruction, properties, if_visit, color_depth,
                recolorable_instructions);
          } else {
            return PropagateFromUsersAndOperands(
                instruction, properties, if_visit, depth, topological_index);
          }
        }();
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
  bool changed;
  HloPassPipeline preprocess_pipeline("coloring_preprocess");
  preprocess_pipeline.AddPass<MpmdInlineExplicitTasks>(partition_);
  preprocess_pipeline.AddPass<MpmdInsertRootTupleShardings>(partition_);
  preprocess_pipeline.AddPass<MpmdComputeAssignedColors>(partition_);
  preprocess_pipeline.AddPass<MpmdUnpackOptimizationBarrier>();
  TF_RETURN_IF_ERROR(preprocess_pipeline.Run(module).status());

  TF_ASSIGN_OR_RETURN(bool enforce_bijective_tasks,
                      EnforceBijectiveTasks(module->entry_computation()));
  if (enforce_bijective_tasks) {
    color_propagation_priority_ = ColorPropagationPriority::kColorDepth;
  }

  auto properties = InstructionProperties::Create(module);
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
  if (module->entry_computation()->root_instruction()->shape().IsTuple()) {
    for (auto* operand :
         module->entry_computation()->root_instruction()->mutable_operands()) {
      TF_RETURN_IF_ERROR(color_root(operand));
    }
  } else {
    TF_RETURN_IF_ERROR(
        color_root(module->entry_computation()->root_instruction()));
  }

  HloPassPipeline postprocess_pipeline("coloring_postprocess");
  postprocess_pipeline.AddPass<MpmdRepackOptimizationBarrier>();
  TF_RETURN_IF_ERROR(postprocess_pipeline.Run(module).status());

  // for now, assume coloring always changes the module
  return true;
}

}  // namespace xla
