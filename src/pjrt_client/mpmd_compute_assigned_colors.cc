/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_compute_assigned_colors.h"

#include <optional>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

absl::StatusOr<std::optional<std::string>>
MpmdComputeAssignedColors::CheckForExplicitShardingColor(
    HloInstruction* instruction, const InstructionProperties& properties) {
  auto explicit_sharding = [&]() -> std::optional<HloSharding> {
    if (instruction->has_sharding()) {
      return instruction->sharding();
    }
    return std::nullopt;
  }();

  if (!explicit_sharding.has_value()) {
    return std::nullopt;
  }

  if (explicit_sharding->IsReplicated()) {
    // make sure this is not a parameter that has been explicitly assigned
    // replicated sharding
    std::optional<int64_t> parameter_number =
        properties.ParameterNumber(instruction);
    if (parameter_number.has_value()) {
      if (AllowOverride(*parameter_number,
                        instruction->parent()
                            ->parent()
                            ->config()
                            .allow_spmd_sharding_propagation_to_parameters())) {
        return std::nullopt;
      }
    } else {
      return std::nullopt;
    }
  }

  zuku::DeviceList devices = [&] {
    if (explicit_sharding->IsReplicated()) {
      return partition_->Devices();
    }
    return *CreateDeviceList(explicit_sharding->tile_assignment());
  }();

  auto devices_color = partition_->FindColor(devices);
  if (!devices_color.has_value()) {
    TF_ASSIGN_OR_RETURN(devices_color,
                        partition_->AllocateColor("sharding", devices, {}));
  }
  VLOG(5) << instruction->name() << " assigned from sharding "
          << *explicit_sharding << " to color=" << *devices_color;

  return std::move(devices_color);
}

absl::Status MpmdComputeAssignedColors::ComputeAssignedColors(
    HloComputation* computation, const InstructionProperties& properties) {
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction == computation->root_instruction() &&
        instruction->opcode() == HloOpcode::kTuple) {
      continue;
    }

    auto color = Color(instruction);
    if (color.has_value()) {
      // make sure the color is allocated in the HLO partition
      if (!partition_->HasColor(*color)) {
        return InvalidArgumentStrCat("color ", *color,
                                     " has not been assigned a device mesh");
      }
    }

    if (!color.has_value() && instruction->opcode() == HloOpcode::kTuple) {
      ColorTuple(instruction);
      continue;
    }

    if (!color.has_value()) {
      TF_ASSIGN_OR_RETURN(color,
                          partition_->ComputeMetadataNameColor(instruction));
      if (color.has_value()) {
        VLOG(5) << "computed metadata name color for " << instruction->name()
                << ",color=" << *color
                << ",metadata=" << instruction->metadata().op_name();
      }
    }

    if (!color.has_value()) {
      TF_ASSIGN_OR_RETURN(
          color, CheckForExplicitShardingColor(instruction, properties));
    } else if (ShardingHasTileAssignment(instruction)) {
      const int64_t num_color_devices =
          partition_->DevicesForColor(*color).size();
      const int64_t num_sharding_devices =
          instruction->sharding().tile_assignment().num_elements();
      if (num_color_devices != num_sharding_devices) {
        return InvalidArgumentStrCat(
            "instruction ", instruction->name(), " was added to color ", *color,
            " with ", num_color_devices, " devices, but sharding ",
            instruction->sharding().ToString(), " is sharded over ",
            num_sharding_devices, "devices");
      }
    }

    if (color.has_value()) {
      AssignColor(instruction, *color);
      properties.ForEachAlias(instruction, [&](HloInstruction* i) {
        if (!IsAssignedColor(i)) {
          AssignColor(i, *color);
        }
      });
    }

    if (instruction->opcode() == HloOpcode::kWhile) {
      TF_RETURN_IF_ERROR(ComputeAssignedColors(
          instruction->called_computations()[0], properties));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> MpmdComputeAssignedColors::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto properties = InstructionProperties::Create(module);

  TF_RETURN_IF_ERROR(
      ComputeAssignedColors(module->entry_computation(), properties));

  // for now, assume coloring always changes the module
  return true;
}

}  // namespace xla
