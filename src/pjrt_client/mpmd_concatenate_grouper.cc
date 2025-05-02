/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_concatenate_grouper.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mpmd_concatenate_grouper.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"

namespace xla {

absl::StatusOr<bool> MpmdConcatenateGrouper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    // if a single reduce user of size 1 on a single dimension
    if (instruction->opcode() == HloOpcode::kConcatenate &&
        instruction->users().size() == 1 &&
        instruction->users().front()->opcode() == HloOpcode::kReduce &&
        ShapeUtil::ElementsIn(instruction->users().front()->shape()) == 1 &&
        instruction->dimensions().size() == 1) {
      std::vector<std::string> colors;
      absl::flat_hash_map<std::string, absl::InlinedVector<HloInstruction*, 8>>
          inputs;
      for (auto* operand : instruction->mutable_operands()) {
        auto operand_color = ColorOrDefault(operand);
        if (!inputs.contains(operand_color)) {
          colors.push_back(operand_color);
        }
        inputs[operand_color].push_back(operand);
      }

      std::vector<HloInstruction*> sub_concatenates;
      const int64_t concatenate_dim = instruction->dimensions().front();
      for (const auto& color : colors) {
        ShapeProto new_shape = inputs[color].front()->shape().ToProto();
        new_shape.mutable_dimensions()->Set(
            concatenate_dim,
            inputs[color].size() * new_shape.dimensions(concatenate_dim));

        auto* sub = module->entry_computation()->AddInstruction(
            HloInstruction::CreateConcatenate(Shape{new_shape}, inputs[color],
                                              concatenate_dim));
        sub_concatenates.push_back(sub);
        AssignColor(sub, color);
      }

      auto* aggregate = module->entry_computation()->AddInstruction(
          HloInstruction::CreateConcatenate(instruction->shape(),
                                            sub_concatenates, concatenate_dim));
      PropagateProperties(instruction, aggregate);

      TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(aggregate));

      TF_RETURN_IF_ERROR(
          module->entry_computation()->RemoveInstruction(instruction));

      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
