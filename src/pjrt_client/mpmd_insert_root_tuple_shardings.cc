/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_insert_root_tuple_shardings.h"

#include <optional>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

absl::StatusOr<bool> MpmdInsertRootTupleShardings::HandleRootTupleShardings(
    HloModule* module) {
  auto* root = module->entry_computation()->root_instruction();
  if (!root->has_sharding() || !root->shape().IsTuple()) {
    return false;
  }

  bool changed = false;
  for (int64_t index = 0; index < root->operand_count(); ++index) {
    const HloSharding& sharding = root->sharding().tuple_elements()[index];
    auto* operand = root->mutable_operand(index);
    const bool allow_sharding_overwrite = AllowOverride(
        index, module->config().allow_spmd_sharding_propagation_to_output());
    if (!sharding.IsReplicated() && !operand->has_sharding()) {
      operand->set_sharding(sharding);
      changed = true;
    }

    if (!allow_sharding_overwrite) {
      VLOG(5) << operand->name() << " is root operand " << index
              << ", which must use fixed sharding " << sharding;
      // we have to create a coloring here to make sure that the correct output
      // sharding is used for this
      HloInstruction* recolor =
          root->parent()->AddInstruction(HloInstruction::CreateCustomCall(
              operand->shape(), {operand}, kCustomCallRootTupleRecolor));
      TF_RETURN_IF_ERROR(root->ReplaceOperandWith(index, recolor));
      std::string color = [&] {
        if (sharding.IsReplicated()) {
          return *partition_->FindOrAllocateGlobalColor();
        }
        zuku::DeviceList devices =
            *CreateDeviceList(sharding.tile_assignment());
        return *partition_->FindOrAllocateColor(devices, nullptr);
      }();
      recolor->set_sharding(sharding);
      AssignColor(recolor, color);
      changed = true;
    }
  }

  return changed;
}

absl::StatusOr<bool> MpmdInsertRootTupleShardings::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // we use sharding in the coloring so we do a (tiny) bit of sharding
  // propagation if a root tuple has explicit sharding annotations, propagate
  // them to the tuple operands
  TF_ASSIGN_OR_RETURN(bool changed, HandleRootTupleShardings(module));

  // for now, assume coloring always changes the module
  return changed;
}

}  // namespace xla
