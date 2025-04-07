/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_insert_reshard.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/legate/legate_sharding.h"
#include "xla/pjrt/legate/mpmd_instruction.h"

namespace xla {

struct ReshardKey {
  zuku::ShardedShape sharded_shape;
  HloInstruction* producer;
};

bool operator==(const ReshardKey& lhs, const ReshardKey& rhs) {
  return lhs.sharded_shape == rhs.sharded_shape && lhs.producer == rhs.producer;
}

template <typename H>
H AbslHashValue(H h, const ReshardKey& key) {
  return H::combine(std::move(h), key.sharded_shape, key.producer);
}

bool NeedsReshard(const zuku::DeviceList& operand_devices,
                  const zuku::DeviceList& devices,
                  const HloSharding& operand_sharding,
                  const HloSharding& sharding) {
  if (operand_sharding.IsReplicated() && sharding.IsReplicated() &&
      operand_devices.Contains(devices)) {
    return false;
  }
  return operand_devices != devices ||
         !MeshEquivalentSharding(operand_sharding, sharding);
}

absl::StatusOr<bool> MpmdInsertReshard::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // at this point, everything should have been flattened into the entry
  // computation
  absl::flat_hash_map<ReshardKey, HloInstruction*> reshard_map;

  HloSharding replicated = HloSharding::Replicate();
  bool changed = false;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      HloComputation* computation = instruction->called_computations()[0];
      auto color = Color(instruction);
      if (!color.has_value()) {
        return InvalidArgumentStrCat("call instruction ", instruction->name(),
                                     " has not been assigned a color");
      }
      auto devices = partition_->DevicesForColor(*color);
      for (int64_t index = 0; index < instruction->operand_count(); ++index) {
        HloInstruction* operand = instruction->mutable_operand(index);
        if (operand->IsCustomCall("DummyLoopOperation") ||
            operand->IsCustomCall("SliceOffset")) {
          continue;
        }
        HloInstruction* parameter = computation->parameter_instruction(index);
        zuku::DeviceList operand_devices = [&] {
          auto color = Color(operand);
          if (color.has_value()) {
            return partition_->DevicesForColor(*color);
          }

          if (operand->opcode() != HloOpcode::kParameter) {
            LOG(FATAL) << "operand " << operand->name()
                       << " has not been assigned a color";
          }
          if (!operand->has_sharding()) {
            return partition_->Devices();
          }
          return GetDevices(operand->sharding(), partition_->Devices());
        }();

        VLOG(5) << operand->name() << " "
                << operand->sharding_or_default(replicated) << " on "
                << operand_devices << " passed to " << parameter->name() << " "
                << parameter->sharding_or_default(replicated) << " on "
                << devices;
        if (NeedsReshard(operand_devices, devices,
                         operand->sharding_or_default(replicated),
                         parameter->sharding_or_default(replicated))) {
          // see if there exists a resharding of this operand
          TF_ASSIGN_OR_RETURN(zuku::ShardedShape parameter_shape,
                              XlaShapeToLegateShape(
                                  parameter->shape(), devices,
                                  parameter->sharding_or_default(replicated)));
          HloInstruction* reshard = [&] {
            auto iter = reshard_map.find({parameter_shape, operand});
            if (iter != reshard_map.end()) {
              return iter->second;
            }
            HloInstruction* reshard =
                module->entry_computation()->AddInstruction(
                    HloInstruction::CreateCustomCall(parameter->shape(),
                                                     {operand}, "Reshard"));
            reshard_map[{parameter_shape, operand}] = reshard;
            reshard->set_sharding(parameter->sharding_ptr());
            AssignColor(reshard, *color);
            return reshard;
          }();
          TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(index, reshard));
          changed = true;
        }
      }
    }
  }

  auto* root = module->entry_computation()->root_instruction();
  if (root->opcode() == HloOpcode::kTuple) {
    for (auto* operand : root->mutable_operands()) {
      if (operand->shape().dimensions().empty()) {
        auto color = Color(operand);
        if (!color.has_value()) {
          return InvalidArgumentStrCat(operand->name(),
                                       " has no assigned color");
        }
        auto devices = partition_->DevicesForColor(*color);
        if (devices.size() < partition_->TotalDevices()) {
          VLOG(5) << "Have output scalar " << operand->name() << " with color "
                  << *color << " sharded over a subset of devices " << devices;
          // scalars should be blasted out to everyone
          TF_ASSIGN_OR_RETURN(auto global_color,
                              partition_->FindOrAllocateGlobalColor("scalars"));
          HloInstruction* reshard = module->entry_computation()->AddInstruction(
              HloInstruction::CreateCustomCall(operand->shape(), {operand},
                                               "Reshard"));
          reshard->set_sharding(HloSharding::Replicate());
          AssignColor(reshard, global_color);
          TF_RETURN_IF_ERROR(operand->ReplaceAllUsesWith(reshard));
          changed = true;
        }
      }
    }
  }

  return changed;
}

}  // namespace xla
