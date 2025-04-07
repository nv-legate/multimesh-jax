/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_MPMD_CROSS_TASK_BARRIER_REMOVER_H_
#define XLA_PJRT_LEGATE_MPMD_CROSS_TASK_BARRIER_REMOVER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/util.h"

namespace xla {

// Removes barriers that are no longer required once
// instructions are partitioned into tasks. Code motion
// does not occur across called computations, which makes
// task boundaries de facto optimization barriers.
// This removes optimization barries that have become redundant
// due to task partition boundaries.
class MpmdCrossTaskBarrierRemover : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdCrossTaskBarrierRemover(HloPartition* partition,
                                       bool remove_parameters)
      : partition_(partition), remove_parameters_(remove_parameters) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdCrossTaskBarrierRemover() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-cross-task-barrier-remover";
  }

 private:
  absl::StatusOr<bool> Visit(HloComputation* computation);

  HloPartition* partition_;
  bool remove_parameters_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_CROSS_TASK_BARRIER_REMOVER_Hß
