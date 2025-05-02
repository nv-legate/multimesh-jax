/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_SHARD_MAP_LOOP_REDUCE_H_
#define XLA_PJRT_MULTIMESH_MPMD_SHARD_MAP_LOOP_REDUCE_H_

#include <memory>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

// Lowers reduces, dots, and custom SPMD partitioning instructions
// that are sharded over contracting dimensions into partitioned, manual shard
// shapes with collectives inserted.  This pass does not change the semantics
// of any operations and normally just involves "pre-fetching" some pieces
// of the SPMD partitioning pass that will be useful for later
// optimization passes.
class MpmdShardMapLoopReduce : public HloModulePass {
 public:
  explicit MpmdShardMapLoopReduce(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdShardMapLoopReduce() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-shard-map-loop-reduce";
  }

 private:
  absl::StatusOr<bool> VisitLoop(HloInstruction* loop,
                                 const InstructionProperties& properties,
                                 HloPassCleanup& cleanup);

  absl::StatusOr<std::unique_ptr<HloModule>> CreateOutlinedPartitionedModule(
      HloInstruction* instruction);

  absl::Status ReplaceWithOutlinedPartitionedModule(HloInstruction* instruction,
                                                    const HloModule& module,
                                                    HloPassCleanup& cleanup);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_SHARD_MAP_LOOP_REDUCE_H_
