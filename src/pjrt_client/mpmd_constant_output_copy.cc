/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_constant_output_copy.h"

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<bool> MpmdConstantOutputCopy::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto* root = module->entry_computation()->root_instruction();
  absl::flat_hash_map<HloInstruction*, absl::InlinedVector<int64_t, 2>>
      root_inputs;
  if (root->shape().IsTuple()) {
    int64_t index = 0;
    for (auto* operand : root->mutable_operands()) {
      root_inputs[operand].push_back(index++);
    }
  } else {
    root_inputs[root].push_back(0);
  }

  auto properties = InstructionProperties::Create(module);

  absl::flat_hash_map<HloInstruction*, int64_t> instruction_depth;
  bool changed = false;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (properties.DerivedInput(instruction) &&
        root_inputs.contains(instruction)) {
      auto* copy = module->entry_computation()->AddInstruction(
          HloInstruction::CreateUnary(instruction->shape(), HloOpcode::kCopy,
                                      instruction));
      auto color = Color(instruction);
      if (color.has_value()) {
        AssignColor(copy, *std::move(color));
      }
      if (instruction == root) {
        module->entry_computation()->set_root_instruction(copy);
      } else {
        for (int64_t index : root_inputs[instruction]) {
          TF_RETURN_IF_ERROR(root->ReplaceOperandWith(index, copy));
        }
      }

      if (instruction->opcode() == HloOpcode::kParameter &&
          instruction->has_sharding()) {
        copy->set_sharding(instruction->sharding_ptr());
      }
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
