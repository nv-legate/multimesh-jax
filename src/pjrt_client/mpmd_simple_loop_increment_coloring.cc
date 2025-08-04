/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_simple_loop_increment_coloring.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

namespace {

constexpr int64_t kMaxInstructionsInTree = 10;
constexpr int64_t kMinInstructionsToVisit = 256;

auto GetIncrementTree(HloInstruction* instruction, HloInstruction* arg_tuple) {
  absl::InlinedVector<HloInstruction*, kMaxInstructionsInTree> tree;
  absl::InlinedVector<HloInstruction*, kMaxInstructionsInTree> to_visit = {
      instruction};
  absl::flat_hash_set<HloInstruction*> visited;

  while (!to_visit.empty()) {
    HloInstruction* next = to_visit.back();
    to_visit.pop_back();
    for (auto* operand : next->mutable_operands()) {
      if (operand != arg_tuple && operand->opcode() != HloOpcode::kConstant) {
        if (operand->has_called_computations()) {
          tree.clear();
          return tree;
        }
        to_visit.push_back(operand);
      }
    }
    if (tree.size() == kMaxInstructionsInTree) {
      tree.clear();
      return tree;
    }
    tree.push_back(next);
  }
  return tree;
}

}  // namespace

absl::StatusOr<bool> MpmdSimpleLoopIncrementColoring::VisitLoop(
    HloInstruction* instruction) {
  bool changed = false;
  auto* computation = instruction->called_computations()[0];
  auto* arg_tuple = computation->parameter_instruction(0);
  std::optional<std::string> increment_color{};
  for (auto* root : computation->root_instruction()->mutable_operands()) {
    if (root->opcode() != HloOpcode::kGetTupleElement ||
        root->operand(0) != arg_tuple) {
      auto tree = GetIncrementTree(root, arg_tuple);
      if (!tree.empty()) {
        VLOG(5) << "loop output " << root->name() << ":" << root->shape()
                << " is a simple increment";
        if (!increment_color.has_value()) {
          TF_ASSIGN_OR_RETURN(increment_color,
                              partition_->AllocateLoopIncrementColor());
        }
        for (auto* instruction : tree) {
          AssignColor(instruction, *increment_color);
        }
        changed = true;
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> MpmdSimpleLoopIncrementColoring::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (module->instruction_count() < kMinInstructionsToVisit) {
    return false;
  }
  std::vector<HloComputation*> to_visit{module->entry_computation()};
  bool changed = false;
  auto properties = InstructionProperties::Create(module);
  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      for (auto* computation : instruction->called_computations()) {
        to_visit.push_back(computation);
      }

      if (instruction->opcode() == HloOpcode::kWhile) {
        TF_ASSIGN_OR_RETURN(bool loop_changed, VisitLoop(instruction));
        changed |= loop_changed;
      }
    }
  }
  return changed;
}

}  // namespace xla
