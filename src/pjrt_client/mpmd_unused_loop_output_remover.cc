/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_unused_loop_output_remover.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {
namespace {

constexpr char kDummyOpCustomCallTarget[] = "DummyLoopOperation";

const HloInstruction* Dealias(const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kGetTupleElement: {
      if (instruction->operand(0)->opcode() ==
          HloOpcode::kOptimizationBarrier) {
        return Dealias(
            instruction->operand(0)->operand(instruction->tuple_index()));
      }
    }
    default:
      return instruction;
  }
}

bool IsRepeatedUseParameter(const HloInstruction* instruction) {
  const HloInstruction* dealiased = Dealias(instruction);
  return dealiased->opcode() == HloOpcode::kGetTupleElement &&
         dealiased->operand(0)->opcode() == HloOpcode::kParameter;
}

}  // namespace

absl::StatusOr<bool> MpmdUnusedLoopOutputRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloComputation*> to_visit = {module->entry_computation()};
  std::vector<HloInstruction*> while_loop_dfs;

  absl::flat_hash_map<HloComputation*, HloInstruction*> parent_loops;
  absl::flat_hash_map<HloComputation*, bool> visiting;

  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    if (visiting[computation]) {
      HloInstruction* parent_loop = parent_loops[computation];
      if (parent_loop) {  // might visit non-loops in the dfs
        while_loop_dfs.push_back(parent_loop);
      }
      to_visit.pop_back();
      continue;
    }

    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile &&
          instruction->has_backend_config()) {
        to_visit.push_back(instruction->called_computations()[0]);
        parent_loops[instruction->called_computations()[0]] = instruction;
      }
    }
    visiting[computation] = true;
  }

  //
  bool changed = false;
  for (HloInstruction* loop : while_loop_dfs) {
    HloComputation* body = loop->called_computations()[0];
    HloInstruction* body_root = body->root_instruction();
    absl::flat_hash_set<int64_t> outputs_used;
    for (auto* user : loop->users()) {
      outputs_used.insert(user->tuple_index());
    }

    std::vector<int64_t> unused_output_indices;
    HloInstruction* input_tuple = loop->mutable_operand(0);
    for (int64_t index = 0; index < input_tuple->operand_count(); ++index) {
      HloInstruction* unused_root = body_root->mutable_operand(index);
      if (!outputs_used.contains(index) &&
          (!IsAssignedColor(unused_root) ||
           !IsRepeatedUseParameter(unused_root))) {
        changed = true;
        // it's very difficult to change the shape of the while-loop
        // for now, we can replace with a dummy op, which makes it easier to
        // change the loop later
        HloInstruction* dummy =
            body->AddInstruction(HloInstruction::CreateCustomCall(
                input_tuple->operand(index)->shape(), {},
                kDummyOpCustomCallTarget));

        VLOG(5) << unused_root->name() << " " << unused_root->shape()
                << " is an unused loop output at index " << index;
        TF_RETURN_IF_ERROR(body_root->ReplaceOperandWith(index, dummy));
        if (unused_root->users().empty()) {
          TF_RETURN_IF_ERROR(
              body->RemoveInstructionAndUnusedOperands(unused_root));
        }
      }
    }

    HloInstruction* body_arg_tuple = body->parameter_instruction(0);
    absl::flat_hash_set<int64_t> inputs_used;
    for (auto* user : body_arg_tuple->users()) {
      if (user->users().empty()) {
        changed = true;
        VLOG(5) << user->name() << " " << user->shape()
                << " is an unused arg tuple user at index "
                << user->tuple_index();
        TF_RETURN_IF_ERROR(body->RemoveInstruction(user));
      } else {
        inputs_used.insert(user->tuple_index());
      }
    }

    std::vector<int64_t> unused_input_indices;
    for (int64_t index = 0; index < input_tuple->operand_count(); ++index) {
      if (!inputs_used.contains(index)) {
        changed = true;
        unused_input_indices.push_back(index);
        HloInstruction* dummy =
            loop->parent()->AddInstruction(HloInstruction::CreateCustomCall(
                input_tuple->operand(index)->shape(), {},
                kDummyOpCustomCallTarget));
        auto* original_operand = input_tuple->mutable_operand(index);
        VLOG(5) << original_operand->name() << " " << original_operand->shape()
                << " is an unused while input at index " << index;
        TF_RETURN_IF_ERROR(input_tuple->ReplaceOperandWith(index, dummy));
        if (original_operand->users().empty()) {
          TF_RETURN_IF_ERROR(RemoveInstructionBackToParameters(
              input_tuple->parent(), original_operand));
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
