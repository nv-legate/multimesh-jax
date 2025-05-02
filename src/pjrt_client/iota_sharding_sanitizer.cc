/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/iota_sharding_sanitizer.h"

#include "xla/util.h"

namespace xla {
namespace {

absl::StatusOr<HloSharding> ToIotaSharding(const HloSharding& sharding,
                                           const zuku::DeviceList& devices) {
  if (sharding.IsReplicated()) {
    return HloSharding::Replicate();
  }

  if (sharding.IsTuple()) {
    OpSharding op_sharding;
    op_sharding.set_type(OpSharding::TUPLE);
    for (const auto& sharding : sharding.tuple_elements()) {
      TF_ASSIGN_OR_RETURN(auto sub_sharding, ToIotaSharding(sharding, devices));
      *op_sharding.add_tuple_shardings() = sub_sharding.ToProto();
    }
    return HloSharding::FromProto(op_sharding);
  }

  if (sharding.ReplicateOnLastTileDim() &&
      sharding.tile_assignment().dimensions().back() == devices.size()) {
    return HloSharding::Replicate();
  }

  if (sharding.IsTileMaximal()) {
    return HloSharding::AssignDevice(0);
  }

  if (!sharding.tile_assignment().iota().has_value()) {
    // try to convert to iota sharding
    const auto& array = sharding.tile_assignment().array();
    int64_t next = *array.begin();
    for (int64_t dev : array) {
      if (dev != next) {
        return InvalidArgumentStrCat("only support iota sharding, got ",
                                     sharding.ToString());
      }
      ++next;
    }
    // okay, I can cast this to a normal iota
    return HloSharding::IotaTile(sharding.tile_assignment().dimensions());
  }

  OpSharding op_sharding = sharding.ToProto();
  op_sharding.set_iota_offset(0);
  return HloSharding::FromProto(op_sharding);
}

}  // namespace

absl::StatusOr<bool> IotaShardingSanitizer::Visit(HloComputation* comp) {
  bool changed = false;

  for (auto* instruction : comp->instructions()) {
    const zuku::DeviceList devices =
        partition_->DevicesForInstruction(instruction);
    for (auto* subcomp : instruction->called_computations()) {
      TF_ASSIGN_OR_RETURN(bool sub_changed, Visit(subcomp));
      changed |= sub_changed;
    }

    if (instruction->has_sharding()) {
      TF_ASSIGN_OR_RETURN(auto new_sharding,
                          ToIotaSharding(instruction->sharding(), devices));
      VLOG(5) << "sanitizing sharding for " << instruction->name() << ": "
              << instruction->sharding() << " to " << new_sharding;
      // iota and non-iota shardings can compare as equal, but we want to
      // canonicalize everything to iota
      if (new_sharding != instruction->sharding() ||
          (new_sharding.tile_assignment().iota().has_value() &&
           !instruction->sharding().tile_assignment().iota().has_value())) {
        changed = true;
        instruction->set_sharding(new_sharding);
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> IotaShardingSanitizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return Visit(module->entry_computation());
}

}  // namespace xla
