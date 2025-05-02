/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_COMPUTATION_GROUPER_H_
#define XLA_PJRT_MULTIMESH_MPMD_COMPUTATION_GROUPER_H_

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

// Groups all instructions with the same partition color into a
// called computation. Instructions unassigned to partitions
// are replicated to any computation where they are needed.
class MpmdComputationGrouper : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdComputationGrouper(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdComputationGrouper() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "mpmd-computation-grouper"; }

 private:
  // Recursively visit loops and computations inside the `computation` called
  // from the `parent_call` and group instructions with the same partition color
  // into called computations.
  absl::Status GroupComputationIntoTasks(HloComputation* computation,
                                         InstructionProperties& properties,
                                         HloInstruction* parent_call);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_COMPUTATION_GROUPER_H_
