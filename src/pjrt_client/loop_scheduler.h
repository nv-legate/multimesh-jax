/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_LOOP_SCHEDULER_H_
#define XLA_PJRT_MULTIMESH_LOOP_SCHEDULER_H_

#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_loop.h"

namespace xla {

// Returns a vector of ordering constraints for the `tasks`.
// The `tasks` are given as a 2D array of iter x stage
// matching the loop `config`. Stage numbers matching
// the `tasks` array are identified by belonging to the
// set `critical_stages`.
absl::StatusOr<std::vector<std::vector<HloInstruction*>>> ScheduleLoops(
    const HloPartition& partition, const LoopConfig& config,
    const std::vector<std::vector<HloInstruction*>>& tasks);

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_LOOP_SCHEDULER_H_
