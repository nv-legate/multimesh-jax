/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_unpack_optimization_barrier.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

namespace {

absl::Status AssignOptimizationBarrierName(HloInstruction* instruction,
                                           HloInstruction* original) {
  if (instruction->has_backend_config()) {
    return absl::AlreadyExistsError(
        "Instruction already has original optimization barrier name");
  }
  // Should be ok to directly assign the name since we are not expecting
  // any other users of the instruction
  instruction->set_raw_backend_config_string(std::string(original->name()));
  return absl::OkStatus();
}

}  // namespace

absl::Status MpmdUnpackOptimizationBarrier::UnpackOptBarrierInstruction(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kOptimizationBarrier) {
    return absl::InvalidArgumentError(
        "Expected optimization barrier instruction");
  }

  HloInstruction* operand = instruction->mutable_operand(0);
  if (operand->opcode() != HloOpcode::kTuple) {
    HloInstruction* unpacked =
        instruction->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            operand->shape(), {operand},
            kCustomCallUnpackedOptimizationBarrier));
    TF_RETURN_IF_ERROR(
        instruction->parent()->ReplaceInstruction(instruction, unpacked));
    TF_RETURN_IF_ERROR(AssignOptimizationBarrierName(unpacked, instruction));
  } else {
    for (int64_t i = 0; i < operand->operand_count(); ++i) {
      HloInstruction* mut_operand = operand->mutable_operand(i);
      HloInstruction* unpacked = instruction->parent()->AddInstruction(
          HloInstruction::CreateCustomCall(
              mut_operand->shape(), {mut_operand},
              kCustomCallUnpackedOptimizationBarrier));
      for (auto* user : instruction->users()) {
        if (user->opcode() != HloOpcode::kGetTupleElement) {
          return absl::InternalError(
              "Optimization barrier with tuple "
              "operand has a non "
              "get-tuple-element user");
        }
        if (user->tuple_index() == i) {
          TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(unpacked));
          TF_RETURN_IF_ERROR(instruction->parent()->RemoveInstruction(user));
        }
      }
      TF_RETURN_IF_ERROR(AssignOptimizationBarrierName(unpacked, instruction));
    }
    TF_RETURN_IF_ERROR(instruction->parent()->RemoveInstruction(instruction));
    if (operand->user_count() == 0) {
      TF_RETURN_IF_ERROR(operand->parent()->RemoveInstruction(operand));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> MpmdUnpackOptimizationBarrier::UnpackComputation(
    HloComputation* computation) {
  bool changed = false;
  for (auto* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kOptimizationBarrier) {
      TF_RETURN_IF_ERROR(UnpackOptBarrierInstruction(instruction));
      changed = true;
    }

    if (instruction->opcode() == HloOpcode::kWhile) {
      TF_ASSIGN_OR_RETURN(
          bool loop_changed,
          UnpackComputation(instruction->called_computations()[0]));
      changed |= loop_changed;
    }
  }

  return changed;
}

absl::StatusOr<bool> MpmdUnpackOptimizationBarrier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      UnpackComputation(module->entry_computation()));

  return changed;
}

}  // namespace xla