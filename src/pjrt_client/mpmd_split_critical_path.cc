/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_split_critical_path.h"

#include <optional>

#include "mpmd_utils.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/critical_path_finder.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/pjrt/multimesh/task_splitter.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/util.h"

namespace xla {

namespace {

bool IsMicrobatchLoop(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kWhile &&
         instruction->has_backend_config();
}

}  // namespace

absl::Status MpmdSplitCriticalPath::SplitComputation(
    HloInstruction* call_to_split, HloComputation* parent_computation,
    uint64_t critical_root_id, const DependencyHloOrdering& ordering) {
  HloComputation* computation = call_to_split->to_apply();
  if (computation->root_instruction()->opcode() != HloOpcode::kTuple) {
    return absl::InvalidArgumentError(
        "Root instruction to split along is not a tuple");
  } else if (critical_root_id >=
             computation->root_instruction()->operand_count()) {
    return absl::InvalidArgumentError("Critical root id is out of bounds");
  }

  const std::string original_color = *Color(call_to_split);

  const auto& options = partition_->GetTaskOptions(original_color);
  if (!options.split_backprop.has_value()) {
    return InvalidArgumentStrCat("color ", original_color,
                                 " is not configured for split backprop");
  }
  std::string critical_suffix = options.split_backprop->first;
  std::string non_critical_suffix = options.split_backprop->second;

  HloInstruction* critical_root =
      computation->root_instruction()->mutable_operand(critical_root_id);

  std::vector<HloInstruction*> critical_instructions;
  std::vector<HloInstruction*> non_critical_instructions;

  // Populate critical and non-critical instructions
  for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
    // Fused root tuple is irrelevant
    if (instruction->IsRoot()) continue;
    // Ordering ExecutesBefore is not reflexive
    if (ordering.ExecutesBefore(instruction, critical_root) ||
        instruction == critical_root) {
      critical_instructions.push_back(instruction);
    } else {
      non_critical_instructions.push_back(instruction);
    }
  }

  TF_ASSIGN_OR_RETURN(
      auto new_colors,
      SplitTask(call_to_split,
                {std::move(critical_instructions),
                 std::move(non_critical_instructions)},
                {critical_suffix, non_critical_suffix}, partition_));

  partition_->SetColorAsNonCritical(new_colors[1]);
  return absl::OkStatus();
}

absl::StatusOr<bool> MpmdSplitCriticalPath::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloComputation* entry_computation = module->entry_computation();
  const DependencyHloOrdering ordering{module};

  HloInstruction* while_instruction = nullptr;
  for (HloInstruction* instruction :
       entry_computation->MakeInstructionPostOrder()) {
    if (IsMicrobatchLoop(instruction)) {
      while_instruction = instruction;
      break;
    }
  }

  // No while loop found, nothing to do
  if (while_instruction == nullptr) {
    return false;
  }

  HloComputation* while_computation =
      while_instruction->called_computations()[0];
  absl::flat_hash_map<const HloInstruction*, int64_t> depth_map =
      ComputeDepthMap(while_computation);

  for (HloInstruction* instruction :
       while_computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      std::optional<std::string> color = Color(instruction);
      if (!color.has_value()) {
        return absl::InvalidArgumentError("Call has no color: " +
                                          instruction->ToString());
      }
      if (partition_->GetTaskOptions(*color).split_backprop.has_value()) {
        TF_ASSIGN_OR_RETURN(
            uint64_t critical_root_id,
            CriticalRootTupleIndex(instruction, while_computation, depth_map));
        VLOG(5) << "Splitting call: " << instruction->name()
                << " with critical root id: " << critical_root_id;
        TF_RETURN_IF_ERROR(SplitComputation(instruction, while_computation,
                                            critical_root_id, ordering));
      }
    }
  }

  return true;
}

}  // namespace xla
