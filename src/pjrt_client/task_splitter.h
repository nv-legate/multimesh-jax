/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_TASK_SPLITTER_H_
#define XLA_PJRT_MULTIMESH_TASK_SPLITTER_H_

#include "absl/status/status.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

// Splits a given call instruction `task` into multiple
// calls based on disjoint groups given by `instructions`.
// Each new called computation should be given a unique
// name corresponding to `suffixes`. New colors for each
// each comnputation will also be added to the `partition`
// based on the `suffixes`.
absl::StatusOr<std::vector<std::string>> SplitTask(
    HloInstruction* task,
    const std::vector<std::vector<HloInstruction*>>& instructions,
    const std::vector<std::string>& suffixes, HloPartition* partition);

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_TASK_SPLITTER_H_
