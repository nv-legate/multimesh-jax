/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_SIMPLE_LOOP_INCREMENT_COLORING_H_
#define XLA_PJRT_MULTIMESH_MPMD_SIMPLE_LOOP_INCREMENT_COLORING_H_

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"

namespace xla {

class MpmdSimpleLoopIncrementColoring : public HloModulePass {
 public:
  explicit MpmdSimpleLoopIncrementColoring(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdSimpleLoopIncrementColoring() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-simple-loop-increment-coloring";
  }

 private:
  absl::StatusOr<bool> VisitLoop(HloInstruction* instruction);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_SIMPLE_LOOP_INCREMENT_COLORING_H_
