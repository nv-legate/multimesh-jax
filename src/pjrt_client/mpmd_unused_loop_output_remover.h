/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_UNUSED_LOOP_OUTPUT_REMOVER_H_
#define XLA_PJRT_MULTIMESH_MPMD_UNUSED_LOOP_OUTPUT_REMOVER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"

namespace xla {

class MpmdUnusedLoopOutputRemover : public HloModulePass {
 public:
  explicit MpmdUnusedLoopOutputRemover(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdUnusedLoopOutputRemover() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-unused-loop-output-remover";
  }

 private:
  absl::StatusOr<bool> Visit(HloComputation* computation);

  absl::StatusOr<bool> SanitizeLoop(HloComputation* computation);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_UNUSED_LOOP_OUTPUT_REMOVER_H_
