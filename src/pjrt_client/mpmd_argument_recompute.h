/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_ARGUMENT_RECOMPUTE_H_
#define XLA_PJRT_MULTIMESH_MPMD_ARGUMENT_RECOMPUTE_H_

#include <cstdint>
#include <optional>
#include <string>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/util.h"

namespace xla {

// HLO pass that decides whether it is beneficial to compute
// an intermediate in a producer task, store it, and send to
// the consumer task or if it should just be recomputed
// by the consumer task to avoid synchronization and communication
// costs between the tasks.
class MpmdArgumentRecompute : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh. The `max_recompute_cost_allowed` sets an upper
  // bound for which arguments are inexpensive enough to allow recomputation.
  explicit MpmdArgumentRecompute(
      HloPartition* partition, int64_t max_recompute_cost_allowed,
      std::optional<int64_t> global_mem_to_compute_ratio = std::nullopt);

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdArgumentRecompute() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "mpmd-argument-recompute"; }

 private:
  // Given an `instruction` with a specific `color assignment,
  // determine if the `operand` with a different coloring should
  // be recomputed from arguments instead of being stored and
  // communicated. The `clone_map` contains the mapping of intermediates
  // that have already been cloned to avoid duplicating instructions.
  absl::StatusOr<bool> MaybeRecomputeOperandFromArguments(
      const std::string& color, HloInstruction* instruction,
      HloInstruction* operand, HloComputation* computation,
      absl::flat_hash_map<const HloInstruction*, HloInstruction*>& clone_map,
      InstructionProperties& properties);

  const int64_t max_recompute_cost_allowed_;
  const int64_t global_mem_to_compute_ratio_;
  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_ARGUMENT_RECOMPUTE_H_
