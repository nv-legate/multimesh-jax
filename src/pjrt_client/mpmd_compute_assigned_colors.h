/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_COMPUTE_ASSIGNED_COLORS_H_
#define XLA_PJRT_MULTIMESH_MPMD_COMPUTE_ASSIGNED_COLORS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

// Assigns a coloring to all user annotated instructions in the HLO module
class MpmdComputeAssignedColors : public HloModulePass {
 public:
  explicit MpmdComputeAssignedColors(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdComputeAssignedColors() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-compute-assigned-colors";
  }

 private:
  absl::Status ComputeAssignedColors(HloComputation* computation,
                                     const InstructionProperties& properties);

  // If the given `instruction`, has not been assigned a color based on
  // user annotation, but the instruction does have an explicit sharding
  // then assign the instruction to a partition over the sharding devices
  // since any instruction explicitly assigned to a given device mesh
  // must be assigned a specific partition color and cannot be freely
  // assigned anywhere.
  absl::StatusOr<std::optional<std::string>> CheckForExplicitShardingColor(
      HloInstruction* instruction, const InstructionProperties& properties);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_COMPUTE_ASSIGNED_COLORS_H_
