/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_CONCATENATE_GROUPER_
#define XLA_PJRT_MULTIMESH_MPMD_CONCATENATE_GROUPER_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"

namespace xla {

// First groups and reduces a concatenate within a computation
// before concatenating and reducing across operands with a
// different partition color. Given the pattern:
//
// a = ... color(red)
// b = ... color(red)
// c = ... color(blue)
// d = ... color(blue)
// e = concatante(a,b,c,d)
// f = reduce(e)
//
// This decomposes the concatenate into:
//
// g = concatenate(a,b)
// h = concatenate(c,d)
// i = reduce(g)
// j = reduce(h)
// k = concantenate(j,j)
// l = reduce(i,j)
class MpmdConcatenateGrouper : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdConcatenateGrouper(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdConcatenateGrouper() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "mpmd-concatenate-grouper"; }

 private:
  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_CONCATENATE_GROUPER_
