/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_cross_task_barrier_remover.h"

#include <cstdint>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

absl::StatusOr<bool> MpmdCrossTaskBarrierRemover::Visit(
    HloComputation* computation) {
  bool changed = false;
  InstructionProperties properties = InstructionProperties::Create(computation);
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kOptimizationBarrier &&
        instruction->shape().IsTuple()) {
      auto* input_tuple = instruction->mutable_operand(0);

      std::vector<HloInstruction*> new_operands;
      absl::flat_hash_set<HloInstruction*> operands_needed;
      absl::InlinedVector<HloInstruction*, 6> users_to_replace;
      for (auto* user : instruction->users()) {
        auto* operand = input_tuple->mutable_operand(user->tuple_index());
        if (remove_parameters_ && properties.DerivedInput(operand)) {
          VLOG(5) << operand->name() << " is a parameter input to "
                  << input_tuple->name() << ", replacing user " << user->name();
          TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(operand));
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(user));
        } else {
          VLOG(5) << operand->name() << " is a still an operand to "
                  << input_tuple->name();
          new_operands.push_back(operand);
          users_to_replace.push_back(user);
        }
        operands_needed.insert(operand);
      }

      if (new_operands.size() == input_tuple->operand_count()) {
        // no replacement
        continue;
      }

      if (new_operands.size() == 1) {
        // not really an opt-barrier anymore
        TF_RETURN_IF_ERROR(
            users_to_replace.front()->ReplaceAllUsesWith(new_operands.front()));
        TF_RETURN_IF_ERROR(
            computation->RemoveInstruction(users_to_replace.front()));
      } else {
        auto* new_tuple = computation->AddInstruction(
            HloInstruction::CreateTuple(new_operands));
        auto* new_barrier =
            computation->AddInstruction(HloInstruction::CreateUnary(
                new_tuple->shape(), HloOpcode::kOptimizationBarrier,
                new_tuple));

        PropagateProperties(input_tuple, new_tuple);
        PropagateProperties(instruction, new_barrier);

        int64_t new_tuple_index = 0;
        for (auto* user : users_to_replace) {
          auto* operand = input_tuple->mutable_operand(user->tuple_index());
          auto* new_gte =
              computation->AddInstruction(HloInstruction::CreateGetTupleElement(
                  new_barrier, new_tuple_index++));
          VLOG(5) << new_gte->name() << " replacing " << user->name()
                  << " for barrier alias " << operand->name();
          PropagateProperties(user, new_gte);
          TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(new_gte));
          TF_RETURN_IF_ERROR(computation->RemoveInstruction(user));
        }
      }

      absl::InlinedVector<HloInstruction*, 6> to_del;
      for (auto* operand : input_tuple->mutable_operands()) {
        if (!operands_needed.contains(operand)) {
          to_del.push_back(operand);
        }
      }

      TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(input_tuple));
      for (auto* operand : to_del) {
        TF_RETURN_IF_ERROR(
            RemoveInstructionBackToParameters(computation, operand));
      }
      changed = true;
    }
  }

  return changed;
}

absl::StatusOr<bool> MpmdCrossTaskBarrierRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  std::vector<HloComputation*> to_visit = {module->entry_computation()};
  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        to_visit.push_back(instruction->called_computations()[0]);
      } else if (instruction->opcode() == HloOpcode::kCall) {
        TF_ASSIGN_OR_RETURN(bool comp_changed,
                            Visit(instruction->called_computations()[0]));

        changed |= comp_changed;
      }
    }
  }
  return changed;
}

}  // namespace xla
