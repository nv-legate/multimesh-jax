/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <string>
#include "xla/pjrt/legate/mpmd_repack_optimization_barrier.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"

#include "xla/pjrt/legate/mpmd_unpack_optimization_barrier.h"
namespace xla {

namespace {

absl::StatusOr<std::string> OptimizationBarrierName(
    HloInstruction* instruction) {
  if (!instruction->has_backend_config()) {
    return InvalidArgumentStrCat("Optimization barrier placeholder ",
                                 instruction->name(),
                                 " has no original instruction name");
  }
  return instruction->raw_backend_config_string();
}

absl::Status ClearOptimizationBarrierNames(HloComputation* computation) {
  for (auto* instruction : computation->instructions()) {
    if (!instruction->has_backend_config() ||
        !instruction->IsCustomCall(kCustomCallUnpackedOptimizationBarrier)) {
      continue;
    }
    instruction->set_raw_backend_config_string("");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> MpmdRepackOptimizationBarrier::RepackComputation(
    HloComputation* computation) {
  bool changed = false;
  absl::flat_hash_map<std::string, std::vector<HloInstruction*>>
      original_name_to_placeholders;

  for (auto* instruction : computation->instructions()) {
    if (instruction->IsCustomCall(kCustomCallUnpackedOptimizationBarrier)) {
      TF_ASSIGN_OR_RETURN(std::string original_name,
                          OptimizationBarrierName(instruction));
      original_name_to_placeholders[original_name].push_back(instruction);
      changed = true;
    }

    if (instruction->opcode() == HloOpcode::kWhile) {
      TF_ASSIGN_OR_RETURN(
          bool loop_changed,
          RepackComputation(instruction->called_computations()[0]));
      changed |= loop_changed;
    }
  }

  for (auto& [original_name, placeholders] : original_name_to_placeholders) {
    if (placeholders.size() == 1) {
      HloInstruction* opt_barrier =
          computation->AddInstruction(HloInstruction::CreateUnary(
              placeholders[0]->shape(), HloOpcode::kOptimizationBarrier,
              placeholders[0]->mutable_operand(0)));
      TF_RETURN_IF_ERROR(
          computation->ReplaceInstruction(placeholders[0], opt_barrier));
    } else {
      std::vector<HloInstruction*> operands;
      operands.resize(placeholders.size());
      std::transform(placeholders.begin(), placeholders.end(), operands.begin(),
                     [](HloInstruction* instruction) {
                       return instruction->mutable_operand(0);
                     });
      HloInstruction* tuple =
          computation->AddInstruction(HloInstruction::CreateTuple(operands));
      HloInstruction* opt_barrier =
          computation->AddInstruction(HloInstruction::CreateUnary(
              tuple->shape(), HloOpcode::kOptimizationBarrier, tuple));
      for (int64_t i = 0; i < placeholders.size(); ++i) {
        HloInstruction* get_tuple_element = computation->AddInstruction(
            HloInstruction::CreateGetTupleElement(opt_barrier, i));
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(placeholders[i],
                                                           get_tuple_element));
      }
    }

    TF_RETURN_IF_ERROR(ClearOptimizationBarrierNames(computation));
  }

  return changed;
}

absl::StatusOr<bool> MpmdRepackOptimizationBarrier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      RepackComputation(module->entry_computation()));

  return changed;
}

}  // namespace xla