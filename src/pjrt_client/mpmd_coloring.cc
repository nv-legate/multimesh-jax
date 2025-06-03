/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_coloring.h"

#include <optional>
#include "absl/container/btree_map.h"

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

absl::flat_hash_map<const HloInstruction*, int64_t> ComputeDepthMap(
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

absl::btree_map<int64_t, std::string> ComputeColorDepthMap(
    const HloComputation* computation,
    const absl::flat_hash_map<const HloInstruction*, int64_t>& depth) {
  absl::flat_hash_map<std::string, int64_t> color_depth;
  absl::btree_map<int64_t, std::string> depth_to_color;

  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    auto inst_color = Color(instruction);
    if (inst_color.has_value()) {
      color_depth[*inst_color] =
          std::max(value_or(color_depth, *inst_color, int64_t(0)),
                   depth.at(instruction));
    }
  }

  for (const auto& [color, depth] : color_depth) {
    // We should never have any collisions because each
    // instruction has a unique depth and color.
    // So each depth maps to only 1 color.
    depth_to_color[depth] = color;
  }

  return depth_to_color;
}

absl::flat_hash_map<const HloInstruction*, int64_t> CreateTopologicalIndexMap(
    const HloComputation* computation) {
  absl::flat_hash_map<const HloInstruction*, int64_t> instruction_to_topo_index;
  const std::vector<HloInstruction*> topological_order =
      computation->MakeInstructionPostOrder();
  for (int64_t i = 0; i < topological_order.size(); ++i) {
    instruction_to_topo_index[topological_order[i]] = i;
  }
  return instruction_to_topo_index;
}

}  // namespace

bool MpmdColoring::PropagateDirectionally(
    const std::vector<HloInstruction*> postorder, const std::string color,
    const InstructionProperties& properties, FilterVisitFn if_visit,
    const absl::flat_hash_set<HloInstruction*>& fixed_colored_instructions,
    bool propagate_forward) {
  // propagate_forward is true if we are propagating forward, false if we are
  // propagating backward
  bool changed = false;

  auto process_instruction = [&](HloInstruction* instruction,
                                 auto related_instructions) {
    if (ColorOrDefault(instruction) == color) {
      for (auto* related_instruction : related_instructions) {
        VLOG(5) << "propagating color " << color << " from "
                << instruction->name() << " to " << related_instruction->name();
        VLOG(5) << "if_visit=" << if_visit(related_instruction)
                << ", fixed_colored_instructions="
                << fixed_colored_instructions.contains(related_instruction);
        if (if_visit(related_instruction) &&
            !fixed_colored_instructions.contains(related_instruction)) {
          VLOG(5) << "propagating color " << color << " to "
                  << related_instruction->name();
          AssignColor(related_instruction, color);
          properties.ForEachAlias(related_instruction, [&](HloInstruction* i) {
            if (!IsAssignedColor(i) ||
                !fixed_colored_instructions.contains(i)) {
              VLOG(5) << "propagating color " << color << " to " << i->name();
              AssignColor(i, color);
            }
          });
          changed = true;
        }
        if (related_instruction->opcode() == HloOpcode::kTuple) {
          ColorTuple(related_instruction);
        }
      }
    }
  };

  if (propagate_forward) {
    for (auto it = postorder.begin(); it != postorder.end(); ++it) {
      process_instruction(*it, (*it)->users());
    }
  } else {
    for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
      process_instruction(*it, (*it)->operands());
    }
  }

  return changed;
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
  } else {
    VLOG(5) << "invalid priority string, using weight priority for color "
               "propagation.";
    return MpmdColoring::ColorPropagationPriority::kWeight;
  }
}

absl::StatusOr<bool> MpmdColoring::PropagateIf(
    HloComputation* computation, const InstructionProperties& properties,
    FilterVisitFn if_visit) {
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
      if (instruction->opcode() == HloOpcode::kTuple) {
        ColorTuple(instruction);
        continue;
      }

      if (if_visit(instruction) && !IsAssignedColor(instruction)) {
        VLOG(5) << "visiting " << instruction->name()
                << " in color propagation from operands/users";
        propagated |= PropagateFromUsersAndOperands(
            instruction, properties, if_visit, depth, topological_index);
      }
      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_ASSIGN_OR_RETURN(bool comp_propagated,
                            PropagateIf(instruction->called_computations()[0],
                                        properties, if_visit));
        propagated |= comp_propagated;
      }
    }
    changed |= propagated;
  }

  return changed;
}

absl::StatusOr<bool> MpmdColoring::PropagateLoopColorDepth(
    HloComputation* computation, const InstructionProperties& properties,
    FilterVisitFn if_visit) {
  bool changed = false;
  absl::flat_hash_set<HloInstruction*> fixed_colored_instructions;
  HloComputation* while_computation = nullptr;

  // First preprocess pass: collect fixed colored instructions outside the
  // loop and propagate their colors to their aliases inside if applicable.
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (IsAssignedColor(instruction)) {
      fixed_colored_instructions.insert(instruction);
      properties.ForEachAlias(instruction, [&](HloInstruction* i) {
        if (!IsAssignedColor(i)) {
          changed = true;
          AssignColor(i, *Color(instruction));
        }
      });
    }
    if (instruction->opcode() == HloOpcode::kWhile) {
      while_computation = instruction->called_computations()[0];
    }
  }

  if (while_computation == nullptr) {
    return changed;
  }

  // Second preprocess pass: collect fixed colored instructions inside the
  // loop and propagate their colors to their aliases outside if applicable.
  const std::vector<HloInstruction*> while_postorder =
      while_computation->MakeInstructionPostOrder();
  for (auto* while_instruction : while_postorder) {
    if (IsAssignedColor(while_instruction)) {
      fixed_colored_instructions.insert(while_instruction);
      properties.ForEachAlias(while_instruction, [&](HloInstruction* i) {
        if (!IsAssignedColor(i)) {
          AssignColor(i, *Color(while_instruction));
        }
      });
    }
  }

  const absl::flat_hash_map<const HloInstruction*, int64_t> loop_depth =
      ComputeDepthMap(while_computation);
  const absl::btree_map<int64_t, std::string> depth_to_colors =
      ComputeColorDepthMap(while_computation, loop_depth);

  // Pass 1: Propagate colors forward
  for (auto it = depth_to_colors.begin(); it != depth_to_colors.end(); ++it) {
    const auto& [depth, color] = *it;
    VLOG(5) << "propagating color " << color << " forward";
    changed |=
        PropagateDirectionally(while_postorder, color, properties, if_visit,
                               fixed_colored_instructions, true);
  }

  // Pass 2: Propagate colors backward
  for (auto it = depth_to_colors.rbegin(); it != depth_to_colors.rend(); ++it) {
    const auto& [depth, color] = *it;
    VLOG(5) << "propagating color " << color << " backward";
    changed |=
        PropagateDirectionally(while_postorder, color, properties, if_visit,
                               fixed_colored_instructions, false);
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

  auto properties = InstructionProperties::Create(module);

  TF_ASSIGN_OR_RETURN(
      bool propagated_depth_ordering,
      PropagateLoopColorDepth(
          module->entry_computation(), properties, [](const HloInstruction* i) {
            return i->opcode() != HloOpcode::kConstant &&
                   (!i->shape().IsTuple() ||
                    i->opcode() == HloOpcode::kRngBitGenerator);
          }));

  // first only visit elementwise propagation
  TF_ASSIGN_OR_RETURN(
      bool propagated_elementwise,
      PropagateIf(module->entry_computation(), properties,
                  [](const HloInstruction* i) {
                    return i->opcode() != HloOpcode::kConstant &&
                           !i->shape().IsTuple() &&
                           (i->IsElementwise() ||
                            i->opcode() == HloOpcode::kParameter ||
                            i->opcode() == HloOpcode::kReduce);
                  }));

  // next only visit non-trivial instructions
  TF_ASSIGN_OR_RETURN(
      bool propagated_nontrivial_shapes,
      PropagateIf(module->entry_computation(), properties,
                  [](const HloInstruction* i) {
                    return i->opcode() != HloOpcode::kConstant &&
                           !i->shape().IsTuple() &&
                           ShapeUtil::ElementsIn(i->shape()) > 1;
                  }));

  // now visit everything
  TF_ASSIGN_OR_RETURN(
      bool propagated_any,
      PropagateIf(module->entry_computation(), properties,
                  [](const HloInstruction* i) {
                    return i->opcode() != HloOpcode::kConstant &&
                           (i->opcode() == HloOpcode::kRngBitGenerator ||
                            !i->shape().IsTuple());
                  }));

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
