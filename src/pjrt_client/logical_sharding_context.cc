/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/logical_sharding_context.h"

#include "src/zuku/mesh.h"
#include "xla/pjrt/legate/json_utils.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

absl::StatusOr<LogicalShardingContext> GetLogicalShardingContext(
    const Json::Value& json, const std::string& context,
    const std::vector<int64_t>& devices) {
  TF_ASSIGN_OR_RETURN(
      auto dims, GetTaskValue<std::vector<int64_t>>(json, context, "dims"));
  TF_ASSIGN_OR_RETURN(auto device_axes, GetTaskValue<std::vector<std::string>>(
                                            json, context, "device_axes"));
  TF_ASSIGN_OR_RETURN(
      auto logical_axes,
      (GetTaskValue<
          std::vector<std::pair<std::string, std::string>>>)(json, context,
                                                             "logical_axes"));

  TF_ASSIGN_OR_RETURN(zuku::DeviceList dl, CreateDeviceList(devices));

  return LogicalShardingContext{
      .devices = std::move(dl),
      .dims = std::move(dims),
      .device_axes = std::move(device_axes),
      .logical_axes = std::move(logical_axes),
  };
}

std::ostream& operator<<(std::ostream& os,
                         const LogicalShardingContext& context) {
  os << "devices=" << context.devices << " axes={";
  for (auto&& ax : context.device_axes) {
    os << " " << ax << ",";
  }
  os << " }, logical={";
  for (auto&& ax_pair : context.logical_axes) {
    os << " " << ax_pair.first << ":" << ax_pair.second << ",";
  }
  os << " }, mesh={";
  for (auto&& dim : context.dims) {
    os << " " << dim << ",";
  }
  os << " }";
  if (context.loop_submesh.has_value()) {
    os << ", submesh=" << context.loop_submesh->SubmeshForIteration(0) << "...";
  }
  return os;
}

std::ostream& operator<<(
    std::ostream& os, const std::shared_ptr<LogicalShardingContext>& context) {
  if (context) {
    os << *context;
  } else {
    os << "null";
  }
  return os;
}

}  // namespace xla
