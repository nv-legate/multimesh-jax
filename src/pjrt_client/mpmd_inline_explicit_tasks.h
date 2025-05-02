/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_INLINE_EXPLICIT_TASKS_H_
#define XLA_PJRT_MULTIMESH_MPMD_INLINE_EXPLICIT_TASKS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

// Inline all explicit tasks in the HLO module
class MpmdInlineExplicitTasks : public HloModulePass {
 public:
  explicit MpmdInlineExplicitTasks(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdInlineExplicitTasks() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-inline-explicit-tasks";
  }

 private:
  // For a given `computation`, iterate through and inline all explicit
  // task blocks. This is equivalent to running the CallInliner
  // on the computation and assigning the scope color as a frontend attribute.
  absl::StatusOr<bool> InlineExplicitTasks(HloComputation* computation);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_INLINE_EXPLICIT_TASKS_H_
