/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_logical_to_gspmd_sharding.h"

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mm_sharding.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/pjrt/multimesh/pycallback_types.h"

namespace xla {
namespace {

// Translates a set of `logical_axis_names` for the given `instruction` to a
// phyiscal sharding based on the logical->physical mapping defined in the
// `context`.
absl::StatusOr<OpSharding> LogicalShardingToOpSharding(
    const HloPartition& partition, HloInstruction* instruction,
    const zuku::DeviceList& sharding_devices,
    const multimesh::TaskOptions& context,
    const LogicalShardingAxes& logical_axis_names) {
  IndexVector<int64_t> dims{context.dims.begin(), context.dims.end()};
  int64_t dim_product = 1;
  for (int64_t dim : dims) {
    dim_product *= dim;
  }

  IndexVector<IndexVector<int>> logical_to_device_axes;
  logical_to_device_axes.resize(logical_axis_names.size());

  logical_to_device_axes.reserve(logical_axis_names.size());
  absl::flat_hash_set<int> device_axes_used;

  IndexVector<int> sharding_on_axis(logical_axis_names.size(), 1);
  for (const auto& [logical_ax, device_ax] : context.logical_axes) {
    int logical_axis_index = 0;
    for (const auto& axis : logical_axis_names) {
      for (const auto& logical_name : axis) {
        VLOG(5) << instruction->name() << " comparing logical name "
                << logical_name << " for axis " << logical_axis_index
                << " against " << logical_ax << ", device axis " << device_ax;
        if (logical_name == logical_ax) {
          int device_axis_index = 0;
          bool axis_found = false;
          for (auto&& ax : context.axes) {
            if (ax == device_ax) {
              axis_found = true;
              break;
            }
            ++device_axis_index;
          }

          if (!axis_found) {
            return InvalidArgumentStrCat("unknown device axis '", device_ax,
                                         "' in autosharding");
          }

          int total_sharding_on_axis =
              sharding_on_axis[logical_axis_index] * dims[device_axis_index];

          if (total_sharding_on_axis <=
                  instruction->shape().dimensions(logical_axis_index) &&
              !device_axes_used.contains(device_axis_index)) {
            VLOG(5) << instruction->name() << " logical axis " << logical_ax
                    << " sharding over device axis " << device_ax
                    << ", index=" << device_axis_index
                    << ", size=" << dims[device_axis_index]
                    << ", total_sharding_on_axis=" << total_sharding_on_axis;
            logical_to_device_axes[logical_axis_index].push_back(
                device_axis_index);
            device_axes_used.insert(device_axis_index);
            sharding_on_axis[logical_axis_index] = total_sharding_on_axis;
          }
          break;
        }
      }
      ++logical_axis_index;
    }
  }

  const bool list_all_devices =
      context.devices.size() < partition.Devices().size();

  TF_ASSIGN_OR_RETURN(
      auto op_sharding,
      LogicalToPhysicalSharding(logical_to_device_axes, sharding_devices, dims,
                                list_all_devices));

  if (!op_sharding.tile_assignment_devices().empty() &&
      op_sharding.tile_assignment_devices_size() != dim_product) {
    return InvalidArgumentStrCat(instruction->name(), " has ",
                                 op_sharding.tile_assignment_devices_size(),
                                 " devices in sharding, but dim product is ",
                                 dim_product);
  }

  if (op_sharding.type() == OpSharding::OTHER) {
    // make sure the sharding evenly divides the dimensions
    for (size_t dim = 0; dim < instruction->shape().dimensions_size(); ++dim) {
      if (instruction->shape().dimensions(dim) %
          op_sharding.tile_assignment_dimensions(dim)) {
        TF_ASSIGN_OR_RETURN(auto bad_sharding,
                            HloSharding::FromProto(op_sharding));
        return InvalidArgumentStrCat(bad_sharding.ToString(),
                                     " does not evenly divide shape ",
                                     instruction->shape().ToString(),
                                     " for instruction ", instruction->name());
      }
    }
  }

  if (VLOG_IS_ON(5)) {
    TF_ASSIGN_OR_RETURN(auto sharding, HloSharding::FromProto(op_sharding));
    VLOG(5) << "autosharding " << instruction->name() << ":"
            << instruction->shape() << " -> " << sharding << " over "
            << partition.DevicesForInstruction(instruction);
  }

  return op_sharding;
}

}  // namespace

absl::StatusOr<bool> MpmdLogicalToGSPMDSharding::ApplySharding(
    HloInstruction* instruction, const InstructionProperties& properties) {
  if (!instruction->has_sharding()) {
    std::optional<LogicalShardingAxes> axes = GetAxes(instruction);
    if (axes.has_value()) {
      auto color = Color(instruction);
      if (color.has_value()) {
        VLOG(5) << "trying to apply sharding to " << instruction->name()
                << " on color " << *color;
        const auto& context = partition_->GetTaskOptions(*color);
        if (!context.logical_axes.empty()) {
          TF_ASSIGN_OR_RETURN(
              OpSharding op_sharding,
              LogicalShardingToOpSharding(*partition_, instruction,
                                          partition_->DevicesForColor(*color),
                                          context, *axes));
          TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                              HloSharding::FromProto(op_sharding));
          instruction->set_sharding(hlo_sharding);
          properties.ForEachBackwardAlias(instruction, [&](HloInstruction* i) {
            if (!i->has_sharding()) {
              const int64_t num_devices =
                  partition_->NumDevicesForInstruction(i);
              if (num_devices ==
                  instruction->sharding().tile_assignment().num_elements()) {
                i->set_sharding(instruction->sharding_ptr());
              }
            }
          });
          return true;
        }
      }
    }
  }
  return false;
}

absl::StatusOr<bool> MpmdLogicalToGSPMDSharding::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloComputation*> to_visit = {module->entry_computation()};
  auto properties = InstructionProperties::Create(module);
  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      switch (instruction->opcode()) {
        case HloOpcode::kGetTupleElement:
        case HloOpcode::kTuple:
        case HloOpcode::kConstant:
          break;
        case HloOpcode::kWhile:
        case HloOpcode::kCall: {
          to_visit.push_back(instruction->called_computations()[0]);
          break;
        }
        case HloOpcode::kParameter:
          // don't map shardings in the entry computation
          // only do parameters in the called computations for now
          if (computation == module->entry_computation()) {
            break;
          }
        default: {
          TF_ASSIGN_OR_RETURN(bool instruction_changed,
                              ApplySharding(instruction, properties));
          changed |= instruction_changed;
          break;
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
