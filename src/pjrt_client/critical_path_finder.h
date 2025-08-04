/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_CRITICAL_PATH_FINDER_H_
#define XLA_PJRT_MULTIMESH_CRITICAL_PATH_FINDER_H_

#include "absl/status/status.h"
#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// Finds the ID of the root instruction of a given call instruction
// that is in the critical path of the computation.
absl::StatusOr<uint64_t> CriticalRootTupleIndex(
    HloInstruction* call, HloComputation* computation,
    const absl::flat_hash_map<const HloInstruction*, int64_t>& depth_map);

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_CRITICAL_PATH_FINDER_H_
