/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mpmd_reorder_shard_map_transpose.h"

#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mm_sharding.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/hlo/utils/hlo_sharding_util.h"

namespace xla {

absl::StatusOr<bool> MpmdReorderShardMapTranspose::VisitLoop(
    HloInstruction* loop, HloPassCleanup& cleanup) {
  bool changed = false;
  HloComputation* body = loop->called_computations()[0];

  for (auto* instruction : body->MakeInstructionPostOrder()) {
    if (instruction->IsCustomCall("SPMDShardToFullShape") &&
        instruction->user_count() == 1 &&
        instruction->users()[0]->opcode() == HloOpcode::kTranspose &&
        Color(instruction) == Color(instruction->users()[0]) &&
        instruction->operand_count() == 1 &&
        instruction->operand(0)->opcode() == HloOpcode::kDot) {
      // found the transpose(shard-map(dot(...))) pattern
      HloInstruction* dot_instr = instruction->mutable_operand(0);
      HloTransposeInstruction* tr_instr =
          static_cast<HloTransposeInstruction*>(instruction->users()[0]);

      // create the new transpose instruction (appropriate, shape, sharding, and
      // color)
      const Shape& dot_shape = dot_instr->shape();
      absl::Span<const int64_t> tr_dims = tr_instr->dimensions();
      TF_ASSIGN_OR_RETURN(
          Shape new_tr_shape,
          ShapeInference::InferTransposeShape(dot_shape, tr_dims));

      HloInstruction* new_tr_instr = body->AddInstruction(
          HloInstruction::CreateTranspose(new_tr_shape, dot_instr, tr_dims));
      new_tr_instr->set_sharding(HloSharding::Manual());

      auto tr_color = Color(tr_instr);
      if (tr_color.has_value()) {
        AssignColor(new_tr_instr, *tr_color);
      }

      // create the shard map instruction (appropriate shape sharding and color)
      HloInstruction* new_shard_map_instr =
          body->AddInstruction(HloInstruction::CreateCustomCall(
              tr_instr->shape(), {new_tr_instr}, "SPMDShardToFullShape"));
      if (!tr_instr->has_sharding()) {
        InvalidArgumentStrCat(
            "instruction ", tr_instr->name(),
            " is user of `SPMDShardToFullShape` but has no sharding");
      }

      // we have to derive the transposed sharding of the original
      // instruction
      new_shard_map_instr->set_sharding(hlo_sharding_util::TransposeSharding(
          instruction->sharding(), tr_instr->dimensions()));

      if (tr_color.has_value()) {
        // tr_instruction and instruction have same color
        AssignColor(new_shard_map_instr, *tr_color);
      }

      TF_RETURN_IF_ERROR(tr_instr->ReplaceAllUsesWith(new_shard_map_instr));

      // cleanup old and unnecessary instructions
      cleanup.RemoveInstruction(tr_instr);
      cleanup.RemoveInstruction(instruction);

      changed = true;
    }
  }

  return changed;
}

absl::StatusOr<bool> MpmdReorderShardMapTranspose::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  HloPassCleanup cleanup;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      TF_ASSIGN_OR_RETURN(bool instruction_changed,
                          VisitLoop(instruction, cleanup));
      changed |= instruction_changed;
    }
  }

  TF_RETURN_IF_ERROR(cleanup.CleanUp());
  return changed;
}

}  // namespace xla
