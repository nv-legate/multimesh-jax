/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_MPMD_REORDER_SHARD_MAP_TRANSPOSE_H_
#define XLA_PJRT_LEGATE_MPMD_REORDER_SHARD_MAP_TRANSPOSE_H_

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

// Finds transpose(shard-map(dot(...))) instructions and reorders
// to shard-map(transpose(dot(...))) so that future alg-simplifier
// passes during the SPMD compiler can optimize the transpose away
class MpmdReorderShardMapTranspose : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdReorderShardMapTranspose() {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdReorderShardMapTranspose() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-reorder-shard-map-transpose";
  }

 private:
  absl::StatusOr<bool> VisitLoop(HloInstruction* loop, HloPassCleanup& cleanup);
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_REORDER_SHARD_MAP_TRANSPOSE_H_