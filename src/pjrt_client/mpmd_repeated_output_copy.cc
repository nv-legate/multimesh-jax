/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_repeated_output_copy.h"

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<bool> MpmdRepeatedOutputCopy::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto* root = module->entry_computation()->root_instruction();
  absl::flat_hash_map<HloInstruction*, absl::InlinedVector<int64_t, 2>>
      root_inputs;
  if (root->shape().IsTuple()) {
    int64_t index = 0;
    for (auto* operand : root->mutable_operands()) {
      VLOG(5) << "found root operand " << operand->name() << " " << operand;
      root_inputs[operand].push_back(index++);
    }
  } else {
    root_inputs[root].push_back(0);
  }

  absl::flat_hash_map<HloInstruction*, int64_t> instruction_depth;
  bool changed = false;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    auto iter = root_inputs.find(instruction);
    if (iter != root_inputs.end() && iter->second.size() > 1) {
      changed = true;
      const auto& root_indices = iter->second;
      for (int64_t output = 1; output < root_indices.size(); ++output) {
        auto* copy = module->entry_computation()->AddInstruction(
            HloInstruction::CreateUnary(instruction->shape(), HloOpcode::kCopy,
                                        instruction));
        TF_RETURN_IF_ERROR(
            root->ReplaceOperandWith(root_indices[output], copy));
      }
    }
  }
  return changed;
}

}  // namespace xla
