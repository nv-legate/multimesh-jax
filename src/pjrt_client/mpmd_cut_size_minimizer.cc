/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_cut_size_minimizer.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mm_sharding.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"

namespace xla {
namespace {

constexpr int kMaxUnaryLookback = 3;

int64_t Weight(const HloInstruction* instruction) {
  return ShapeUtil::ElementsIn(GetSpmdShape(instruction));
}

std::optional<std::string> SingleUserDifferentColor(
    HloComputation* computation, HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kParameter ||
      instruction == computation->root_instruction()) {
    return std::nullopt;
  }

  auto color = Color(instruction);
  if (!color.has_value()) {
    return std::move(color);
  }

  absl::flat_hash_set<std::string> user_colors;
  for (auto* user : instruction->users()) {
    auto user_color = Color(user);
    if (user_color.has_value()) {
      user_colors.insert(*std::move(user_color));
    }
  }

  if (user_colors.size() == 1 && *user_colors.begin() != *color) {
    return *std::move(user_colors.begin());
  }

  return std::nullopt;
}

// The searchs backwards and return a linear use "tree"
// back to the first non-unary operand. If the instruction is not
// a unary op, the tree will consist of a single operation that is
// the input instruction. The last element in the tree will be the
// the first non-unary op.
absl::InlinedVector<HloInstruction*, kMaxUnaryLookback> GetSingleUserUnaryTree(
    HloInstruction* instruction) {
  absl::InlinedVector<HloInstruction*, kMaxUnaryLookback> to_visit = {
      instruction};
  absl::InlinedVector<HloInstruction*, kMaxUnaryLookback> tree;
  while (!to_visit.empty()) {
    HloInstruction* next = to_visit.back();
    to_visit.pop_back();
    tree.push_back(next);
    if (next->users().size() > 1) {
      return tree;
    }
    switch (next->opcode()) {
      case HloOpcode::kTranspose:
      case HloOpcode::kConvert:
      case HloOpcode::kReshape:
      case HloOpcode::kBitcast:
      case HloOpcode::kNegate:
      case HloOpcode::kSqrt:
      case HloOpcode::kCos:
      case HloOpcode::kExp:
      case HloOpcode::kSin:
      case HloOpcode::kNot:
        to_visit.push_back(next->mutable_operand(0));
        break;
      default:
        return tree;
    }
  }
  return tree;
}

bool IsLargerThanOperands(const HloInstruction* instruction) {
  int64_t operand_weight = 0;
  for (auto* operand : instruction->operands()) {
    operand_weight += Weight(operand);
  }
  return Weight(instruction) > operand_weight;
}

}  // namespace

absl::StatusOr<bool> MpmdCutSizeMinimizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloComputation*> to_visit = {module->entry_computation()};
  bool changed = false;

  // first visit instructions where the operands are larger than the output
  // and moving the instruction to colocate with users would decrease data
  // movement
  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();
    auto postorder = computation->MakeInstructionPostOrder();
    for (auto iter = postorder.rbegin(); iter != postorder.rend(); ++iter) {
      auto* instruction = *iter;
      if (instruction->opcode() == HloOpcode::kWhile) {
        to_visit.push_back(instruction->called_computations()[0]);
        continue;
      }

      bool skip = false;
      switch (instruction->opcode()) {
        case HloOpcode::kCustomCall:
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kCopy:
          skip = true;
          break;
        default:
          break;
      }

      if (skip) {
        continue;
      }

      auto instruction_color = Color(instruction);
      if (!instruction_color.has_value()) {
        // this isn't actually assigned anywhere so there's nowhere to move it
        continue;
      }
      auto single_user_color =
          SingleUserDifferentColor(computation, instruction);
      if (single_user_color.has_value()) {
        // TODO: support resharding the argument to different mesh sizes
        if (partition_->EquivalentMesh(*single_user_color,
                                       *instruction_color)) {
          auto unary_tree = GetSingleUserUnaryTree(instruction);
          if (IsLargerThanOperands(unary_tree.back())) {
            changed = true;
            for (auto* element : unary_tree) {
              VLOG(5) << "moving " << instruction->name()
                      << " to colocate with users";
              AssignColor(instruction, *single_user_color);
            }
          }
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
