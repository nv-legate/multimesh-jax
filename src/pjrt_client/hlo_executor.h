/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef XLA_PJRT_HLO_EXECUTOR_H_
#define XLA_PJRT_HLO_EXECUTOR_H_

#include "src/zuku/tiled_array.h"
#include "src/zuku/vector.h"
#include "xla/pjrt/legate/legate_computation.h"
#include "xla/pjrt/legate/scalar_argument.h"

namespace xla {

void RunExecutable(zuku::Stream* zs, int64_t run_id, zuku::DeviceList devices,
                   zuku::Processor p, std::shared_ptr<LegateCompiler> compiler,
                   std::vector<ScalarArgument> scalars,
                   zuku::ro_vector<zuku::ShardedArray> inputs,
                   zuku::rw_vector<zuku::ShardedArray> outputs,
                   const zuku::ArrayTile& temp);

}  // namespace xla

#endif  // XLA_PJRT_HLO_EXECUTOR_H_