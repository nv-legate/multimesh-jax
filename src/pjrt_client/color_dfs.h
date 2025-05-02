/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef XLA_PJRT_MULTIMESH_COLOR_DFS_H_
#define XLA_PJRT_MULTIMESH_COLOR_DFS_H_

#include "absl/status/status.h"
#include "src/zuku/mesh.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

std::vector<HloInstruction*> ColorSortedPostorder(HloComputation* computation);

}  // namespace xla

#endif
