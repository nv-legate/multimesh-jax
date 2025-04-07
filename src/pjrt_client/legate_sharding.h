/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_LEGATE_SHARDING_H_
#define XLA_PJRT_LEGATE_LEGATE_SHARDING_H_

#include "absl/container/inlined_vector.h"
#include "src/zuku/mesh.h"
#include "src/zuku/shape.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/xla_data.pb.h"

namespace xla {

template <class T>
using IndexVector = absl::InlinedVector<T, 6>;

zuku::DeviceList GetDevices(const HloSharding& sharding,
                            zuku::DeviceList default_devices);

bool IsReplicatedOnAssignedSubmesh(const HloSharding& sharding);

bool IsReplicatedOrSubmeshReplicate(const HloSharding& sharding);

bool MeshEquivalentSharding(const HloSharding& lhs, const HloSharding& rhs);

absl::StatusOr<zuku::ShardedShape> CanonicalizeSharding(
    const HloSharding& sharding, const Shape& shape, bool shape_is_global);

absl::StatusOr<HloSharding> ReplicateDims(
    absl::Span<const int64_t> replicated_dims, const HloSharding& sharding);

// The computation here is annoyingly complicated. There are 0..n device axes
// assigned to each logical axis. This defines a logical device grid. Consider
// the logical->device mapping [ (0,2), (), (1,) ] The 1st logical axes is
// sharded over device axes 0,2, the 2nd logical axis is not sharded, the 3rd
// logical axis is sharded over device axis 1. For a device grid [2,2,2], the
// logical tile dims are then tile_dims = [4,1,2] To compute the device
// assignment requires some tedious index math. Logical device 5 -> Logical
// Index [2,0,1] -> Device MultiIndex [ (1,0), (), (1) ]
// -> Device Index [1,1,0] -> Physical device 6
absl::StatusOr<OpSharding> LogicalToPhysicalSharding(
    const IndexVector<IndexVector<int>>& logical_to_device_axes,
    const zuku::DeviceList& devices, const IndexVector<int64_t>& dims,
    bool list_all_devices = false);

absl::StatusOr<OpSharding> LogicalToPhysicalSharding(
    const IndexVector<IndexVector<int>>& logical_to_device_axes,
    std::pair<int64_t, int64_t> slice, const IndexVector<int64_t>& dims,
    bool list_all_devices = false);

// Increase or decrease the size of a sharding to match a tile assignment with a
// given number of elements Returns the new sharding if heuristics can find a
// valid resharding, other nullopt
std::optional<HloSharding> ResizeSharding(
    const Shape& shape, const HloSharding& sharding, int64_t num_elements,
    std::optional<zuku::DeviceList> devices = std::nullopt);

Shape GetSpmdShape(const HloInstruction* instruction);

template <class Container>
Container ComputeIndexSet(int64_t global_index, const Container& dims) {
  Container indices(dims.size());
  int remainder = global_index;
  int dim_stride = 1;
  for (int dim = 0; dim < dims.size(); ++dim) {
    dim_stride *= dims[dim];
  }

  for (size_t dim = 0; dim < dims.size(); ++dim) {
    dim_stride /= dims[dim];
    indices[dim] = remainder / dim_stride;
    remainder -= indices[dim] * dim_stride;
  }

  return indices;
}

template <size_t N, size_t M>
int ComputeGlobalIndex(const absl::InlinedVector<int64_t, N>& indices,
                       const absl::InlinedVector<int64_t, M>& dims) {
  int stride = 1;
  int index = 0;
  for (int dim = indices.size() - 1; dim >= 0; --dim) {
    index += stride * indices[dim];
    stride *= dims[dim];
  }
  return index;
}

absl::StatusOr<zuku::ShardedShape> XlaShapeToLegateShape(
    const Shape& shape, const zuku::DeviceList& devices,
    const HloSharding& sharding, bool shape_is_global = true);

}  // namespace xla

#endif
