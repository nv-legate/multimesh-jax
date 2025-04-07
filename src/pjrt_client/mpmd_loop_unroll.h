/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_MPMD_LOOP_UNROLL_H_
#define XLA_PJRT_LEGATE_MPMD_LOOP_UNROLL_H_

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

// Finds computations with the same color or same device mesh
// and fuses them into a single computation, creating a new set
// of parameter instructions and new set of outputs.
class MpmdLoopUnroll : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdLoopUnroll(HloPartition* partition) : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdLoopUnroll() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "mpmd-loop-unroll"; }

 private:
  absl::Status Unroll(HloComputation* parent, HloComputation* body,
                      HloInstruction* while_loop, HloInstruction* input_tuple,
                      const InstructionProperties& properties,
                      HloPassCleanup& cleanup);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_LOOP_UNROLL_H_
