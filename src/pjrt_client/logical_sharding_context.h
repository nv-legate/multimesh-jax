/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_AUTOSHARD_CONTEXT_H_
#define XLA_PJRT_LEGATE_AUTOSHARD_CONTEXT_H_

#include <vector>

#include "json/json.h"
#include "src/zuku/mesh.h"
#include "xla/pjrt/legate/mpmd_loop.h"

namespace xla {

template <class T>
using InlineVector = absl::InlinedVector<T, 6>;

// Class encapsulating logical names for sharding
// the axes in an instruction. `axes` should match
// the number of dimensions in the instruction shape.
// Each dimension can have multiple logical names
// for matching the sharding pattern
struct LogicalShardingAxes {
  InlineVector<InlineVector<std::string>> axes;
};

// Class encapsulating how to convert logical sharding metadata
// attached to physical sharding on a device mesh
struct LogicalShardingContext {
  zuku::DeviceList devices;
  // The dimension shape for the physical mesh
  std::vector<int64_t> dims;
  // Unique names assigned to each physical mesh dimension
  std::vector<std::string> device_axes;
  // (logical,device) pairs mapping logical axis names to physical mesh names
  std::vector<std::pair</*logical=*/std::string, /*device=*/std::string>>
      logical_axes;
  std::optional<LoopDependentSubmesh> loop_submesh;
};

// Helper function for convering a Json config
// to an autosharding context
// `json` is a json node containing all the requiried subentries
// `context` is a name to help in debugging missing fields
// `devices` is the set of devices for the context
absl::StatusOr<LogicalShardingContext> GetLogicalShardingContext(
    const Json::Value& json, const std::string& context,
    const std::vector<int64_t>& devices);

std::ostream& operator<<(std::ostream& os,
                         const LogicalShardingContext& context);

std::ostream& operator<<(
    std::ostream& os, const std::shared_ptr<LogicalShardingContext>& context);

}  // namespace xla

#endif
