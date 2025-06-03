/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_inplace_collectives.h"

#include <algorithm>
#include <optional>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {
namespace {

std::optional<int64_t> GetMatchingCollectiveInput(HloComputation* computation,
                                                  int64_t output_number) {
  auto* root_input =
      computation->root_instruction()->mutable_operand(output_number);
  bool found_collective = false;
  while (root_input) {
    switch (root_input->opcode()) {
      case HloOpcode::kParameter:
        if (found_collective) {
          return root_input->parameter_number();
        }
        return std::nullopt;
      case HloOpcode::kCustomCall:
        if (!root_input->IsCustomCall("SPMDShardToFullShape") &&
            !root_input->IsCustomCall("SPMDFullToShardShape")) {
          return std::nullopt;
        }
      case HloOpcode::kAllReduce:
        found_collective = true;
      case HloOpcode::kConvert:
        root_input = root_input->mutable_operand(0);
        break;
      default:
        return std::nullopt;
    }
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<bool> MpmdInPlaceCollectives::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      auto* computation = instruction->called_computations()[0];
      for (auto* user : instruction->users()) {
        std::optional<int64_t> collective_operand_index =
            GetMatchingCollectiveInput(computation, user->tuple_index());
        if (collective_operand_index.has_value()) {
          auto* input = instruction->mutable_operand(*collective_operand_index);
          VLOG(5) << "found match: " << input->name() << " -> " << user->name()
                  << " on " << instruction->name();
          if (!input->metadata().scheduling_name().empty()) {
            user->set_metadata_scheduling_name(
                input->metadata().scheduling_name());
            changed = true;
          }
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
