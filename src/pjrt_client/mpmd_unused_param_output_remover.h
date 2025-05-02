/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_UNUSED_PARAM_OUTPUT_REMOVER_H_
#define XLA_PJRT_MULTIMESH_MPMD_UNUSED_PARAM_OUTPUT_REMOVER_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
namespace xla {

class MpmdUnusedParamOutputRemover : public HloModulePass {
 public:
  explicit MpmdUnusedParamOutputRemover(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdUnusedParamOutputRemover() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-unused-param-output-remover";
  }

 private:
  absl::StatusOr<bool> Visit(HloInstruction* call_instruction);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_UNUSED_PARAM_OUTPUT_REMOVER_H_
