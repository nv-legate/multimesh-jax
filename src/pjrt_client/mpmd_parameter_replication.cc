/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_parameter_replication.h"

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"

namespace xla {

absl::StatusOr<bool> MpmdParameterReplication::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  absl::flat_hash_set<HloInstruction*> replicated;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kParameter &&
        instruction->has_sharding() &&
        !instruction->sharding().IsReplicated()) {
      if (ShapeUtil::ElementsIn(instruction->shape()) <= num_elements_cutoff_) {
        if (!instruction->users().empty()) {
          auto* replicated_copy = module->entry_computation()->AddInstruction(
              HloInstruction::CreateUnary(instruction->shape(),
                                          HloOpcode::kCopy, instruction));
          replicated_copy->SetAndSanitizeName(
              absl::StrCat("replicated-", instruction->name()));
          replicated_copy->set_sharding(HloSharding::Replicate());
          TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(replicated_copy));

          TF_ASSIGN_OR_RETURN(std::string color,
                              partition_->FindOrAllocateGlobalColor());
          // Assign the argument to the same color
          VLOG(5) << "created replicated copy of " << instruction->name() << " "
                  << instruction->shape() << " with color=" << color;
          replicated.insert(replicated_copy);
          AssignColor(replicated_copy, std::move(color));
          changed = true;
        }
      } else {
        VLOG(5) << "size of " << instruction->name() << " "
                << instruction->shape() << " is larger than cutoff "
                << num_elements_cutoff_;
      }
    } else if (instruction->opcode() == HloOpcode::kTuple &&
               instruction->users().size() == 1 &&
               instruction->users().front()->opcode() == HloOpcode::kWhile) {
      auto* arg_tuple = instruction->users()
                            .front()
                            ->called_computations()[0]
                            ->parameter_instruction(0);
      for (auto* user : arg_tuple->users()) {
        auto* matching_operand =
            instruction->mutable_operand(user->tuple_index());
        if (replicated.contains(matching_operand)) {
          VLOG(5) << "propagating replication from " << matching_operand->name()
                  << " to loop argument " << user->name();
          user->set_sharding(HloSharding::Replicate());
        }
        replicated.insert(user);
      }
    }
  }
  return changed;
}

}  // namespace xla
