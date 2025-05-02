/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_uniquify_colors.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"

namespace xla {

absl::StatusOr<bool> MpmdUniquifyColors::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // recolor everything to have a unique color per-task
  int64_t next_color = 0;
  absl::flat_hash_map<std::string, std::string> color_map;
  std::vector<HloComputation*> to_visit = {module->entry_computation()};

  bool changed = false;

  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    auto postorder = computation->MakeInstructionPostOrder();
    for (auto* instruction : postorder) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        to_visit.push_back(instruction->called_computations()[0]);
      } else if (instruction->opcode() == HloOpcode::kCall) {
        std::optional<std::string> color = Color(instruction);
        if (!color.has_value()) {
          return InvalidArgumentStrCat(module->name(), " has call ",
                                       instruction->name(),
                                       " without color assigned");
        }
        std::string new_color = *color;
        if (color_map.contains(*color)) {
          new_color = absl::StrCat(*color, ".", next_color);
        }
        color_map[new_color] = *color;

        auto* called_comp = instruction->called_computations()[0];
        if (new_color != *color) {
          changed = true;
          for (auto* sub : called_comp->instructions()) {
            AssignColor(sub, new_color);
          }
          // make sure the get-tuple-element outputs of the call also
          // have the correct color assigned
          for (auto* user : instruction->users()) {
            AssignColor(user, new_color);
          }
          AssignColor(instruction, new_color);
          ++next_color;
        }
      }
    }
    // make sure all param colors are in the recolor map
    for (auto* instruction : postorder) {
      // make sure all parameter colors are in the remap map
      if (instruction->opcode() == HloOpcode::kParameter) {
        std::optional<std::string> color = Color(instruction);
        if (color.has_value()) {
          if (!color_map.contains(*color)) {
            color_map[*color] = *color;
          }
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(partition_->Recolor(color_map));

  return changed;
}

}  // namespace xla
