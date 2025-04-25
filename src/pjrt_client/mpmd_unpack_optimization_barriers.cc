/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_simple_loop_increment_coloring.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

namespace {

absl::Status AssignOptimizationBarrierName(HloInstruction* instruction,
                                           HloInstruction* original) {
  FrontendAttributes attrs = instruction->frontend_attributes();
  if (attrs.map().contains("original_instruction")) {
    return absl::AlreadyExistsError(
        "Instruction already has original optimization barrier name");
  }
  (*attrs.mutable_map())["original_instruction"] = original->name();
  instruction->set_frontend_attributes(std::move(attrs));
  return absl::OkStatus();
}

}  // namespace

absl::Status UnpackOptBarrierInstruction(HloInstruction* instruction) {
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
    TF_RETURN_IF_ERROR(instruction->ReplaceWithNewInstruction(unpacked));
    TF_RETURN_IF_ERROR(AssignOptimizationBarrierName(unpacked, instruction))
  } else {
    for (int64_t i = 0; i < operand->operand_count(); ++i) {
      HloInstruction* mut_operand = operand->mutable_operand(i);
      HloInstruction* unpacked = instruction->parent()->AddInstruction(
          HloInstruction::CreateCustomCall(
              mut_operand->shape(), {mut_operand},
              kCustomCallUnpackedOptimizationBarrier));
      for (auto* user : unpacked->users()) {
        if (user->opcode() != HloOpcode::kGetTupleElement) {
          return absl::InternalError(
              "Optimization barrier with tuple "
              "operand has a non "
              "get-tuple-element user");
        }
        if (user->tuple_index() == i) {
          TF_RETURN_IF_ERROR(instruction->ReplaceUseWith(user, unpacked));
        }
      }
      TF_RETURN_IF_ERROR(AssignOptimizationBarrierName(unpacked, instruction))
    }
    TF_RETURN_IF_ERROR(instruction->parent()->RemoveInstruction(instruction));
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> UnpackComputation(HloComputation* computation) {
  bool changed = false;
  for (auto* instruction : computation->instructions()) {
    TF_RETURN_IF_ERROR(UnpackOptBarrierInstruction(instruction));
    changed = true;

    if (instruction->opcode() == HloOpcode::kWhile) {
      TF_ASSIGN_OR_RETURN(
          bool loop_changed,
          UnpackComputation(instruction->called_computations()[0]));
      changed |= loop_changed;
    }
  }

  return changed;
}

absl::StatusOr<bool> MpmdUnpackOptimizationBarriers::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  TF_ASSIGN_OR_RETURN(bool changed,
                      UnpackComputation(module->entry_computation()));

  return changed;
}

}  // namespace xla
