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
#include "xla/pjrt/legate/mpmd_utils.h"
#include "xla/util.h"

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

HloInstruction* MakeReshardIfNeeded(
    HloComputation* parent,
    absl::flat_hash_map<ReshardKey, HloInstruction*>& reshard_map,
    const zuku::DeviceList& operand_devices, const zuku::DeviceList& devices,
    HloInstruction* operand, HloInstruction* user) {}

absl::StatusOr<bool> MpmdInsertReshard::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // at this point, everything should have been flattened into the entry
  // computation
  absl::flat_hash_map<ReshardKey, HloInstruction*> reshard_map;

  // lambda to get instructions devices
  auto get_instruction_devices = [&](HloInstruction* instruction) {
    auto color = Color(instruction);
    if (color.has_value()) {
      return partition_->DevicesForColor(*color);
    }

    if (instruction->opcode() != HloOpcode::kParameter) {
      LOG(FATAL) << "instruction " << instruction->name()
                 << " has not been assigned a color";
    }
    if (!instruction->has_sharding()) {
      return partition_->Devices();
    }
    return GetDevices(instruction->sharding(), partition_->Devices());
  };

  HloSharding replicated = HloSharding::Replicate();
  auto make_reshard_if_needed = [&](const std::string& user_color,
                                    const zuku::DeviceList& operand_devices,
                                    const zuku::DeviceList& devices,
                                    HloInstruction* operand,
                                    HloInstruction* user) -> HloInstruction* {
    if (NeedsReshard(operand_devices, devices,
                     operand->sharding_or_default(replicated),
                     user->sharding_or_default(replicated))) {
      // see if there exists a resharding of this operand
      zuku::ShardedShape parameter_shape = *std::move(XlaShapeToLegateShape(
          user->shape(), devices, user->sharding_or_default(replicated)));
      HloInstruction* reshard = [&] {
        auto iter = reshard_map.find({parameter_shape, operand});
        if (iter != reshard_map.end()) {
          return iter->second;
        }
        HloInstruction* reshard = module->entry_computation()->AddInstruction(
            HloInstruction::CreateCustomCall(user->shape(), {operand},
                                             kCustomCallReshard));
        reshard_map[{parameter_shape, operand}] = reshard;
        reshard->set_sharding(user->sharding_ptr());
        AssignColor(reshard, user_color);
        return reshard;
      }();
      return reshard;
    }
    return nullptr;
  };

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
        if (operand->IsCustomCall(kCustomCallDummyOperation) ||
            operand->IsCustomCall(kCustomCallSliceOffset)) {
          continue;
        }
        HloInstruction* parameter = computation->parameter_instruction(index);
        zuku::DeviceList operand_devices = get_instruction_devices(operand);

        VLOG(5) << operand->name() << " "
                << operand->sharding_or_default(replicated) << " on "
                << operand_devices << " passed to " << parameter->name() << " "
                << parameter->sharding_or_default(replicated) << " on "
                << devices;

        HloInstruction* reshard = make_reshard_if_needed(
            *color, operand_devices, devices, operand, parameter);
        if (reshard) {
          TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(index, reshard));
          changed = true;
        }
      }
    }
  }

  auto* root = module->entry_computation()->root_instruction();
  if (root->opcode() == HloOpcode::kTuple) {
    for (int index = 0; index < root->operand_count(); index++) {
      auto* operand = root->mutable_operand(index);
      if (operand->IsCustomCall(kCustomCallRootTupleRecolor)) {
        VLOG(5) << operand->name() << " is a custom call for RootTupleRecolor";

        // determine the instruction sharding and devices
        HloInstruction* rtr_instruction = operand;
        auto rtr_color = Color(operand);
        if (!rtr_color.has_value()) {
          return InvalidArgumentStrCat("custom call instruction ",
                                       operand->name(),
                                       " has not been assigned a color");
        }
        auto rtr_devices = partition_->DevicesForColor(*rtr_color);
        // determine the operand sharding and devices
        if (rtr_instruction->operand_count() != 1) {
          LOG(FATAL) << "RootTupleRecolor has "
                     << rtr_instruction->operand_count()
                     << " operands (expected 1)";
        }
        if (rtr_instruction->users().size() != 1) {
          return InternalStrCat("recoloring instruction ",
                                rtr_instruction->name(),
                                " should have a single user (the root tuple)");
        }
        HloInstruction* rtr_operand = rtr_instruction->mutable_operand(0);
        if (rtr_operand->IsCustomCall(kCustomCallDummyOperation) ||
            rtr_operand->IsCustomCall(kCustomCallSliceOffset)) {
          return InternalStrCat("dummy operations/slice offsets ",
                                rtr_operand->name(),
                                " cannot be a root tuple operand");
        }
        zuku::DeviceList rtr_operand_devices =
            get_instruction_devices(rtr_operand);

        HloInstruction* reshard =
            make_reshard_if_needed(*rtr_color, rtr_operand_devices, rtr_devices,
                                   rtr_operand, rtr_instruction);
        if (reshard) {
          TF_RETURN_IF_ERROR(rtr_instruction->ReplaceAllUsesWith(reshard));
        } else {
          TF_RETURN_IF_ERROR(rtr_instruction->ReplaceAllUsesWith(rtr_operand));
        }
        TF_RETURN_IF_ERROR(
            module->entry_computation()->RemoveInstruction(rtr_instruction));
        changed = true;
      } else if (operand->shape().dimensions().empty()) {
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
                              partition_->FindOrAllocateGlobalColor());
          HloInstruction* reshard = module->entry_computation()->AddInstruction(
              HloInstruction::CreateCustomCall(operand->shape(), {operand},
                                               kCustomCallReshard));
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
