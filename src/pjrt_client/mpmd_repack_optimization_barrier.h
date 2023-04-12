/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_REPACK_OPTIMIZATION_BARRIER_H_
#define XLA_PJRT_MULTIMESH_MPMD_REPACK_OPTIMIZATION_BARRIER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class MpmdRepackOptimizationBarrier : public HloModulePass {
 public:
  explicit MpmdRepackOptimizationBarrier() {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdRepackOptimizationBarrier() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-repack-optimization-barrier";
  }

 private:
  absl::StatusOr<bool> RepackComputation(HloComputation* computation);
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_REPACK_OPTIMIZATION_BARRIER_H_
