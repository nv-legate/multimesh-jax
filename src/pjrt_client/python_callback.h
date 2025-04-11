/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_PYTHON_CALLBACK_H_
#define XLA_PJRT_LEGATE_PYTHON_CALLBACK_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_loop.h"
namespace xla {

absl::StatusOr<std::vector<HloInstruction*>> CallCustomPythonCallback(
    const HloPartition& partition, const LoopConfig& config,
    const std::vector<std::vector<HloInstruction*>>& tasks);

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_PYTHON_CALLBACK_H_
