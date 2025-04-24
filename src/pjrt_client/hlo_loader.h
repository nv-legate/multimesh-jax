/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef XLA_PJRT_HLO_LOADER_H_
#define XLA_PJRT_HLO_LOADER_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "legate_computation.h"
#include "src/zuku/processor.h"

namespace xla {

struct HloLoaderOptions {
  bool print_stats = false;
  std::optional<uint32_t> num_partitions = std::nullopt;
  std::optional<uint64_t> hlo_id = std::nullopt;
};

void LoadAndCompile(int64_t run_id, zuku::Processor p,
                    const std::shared_ptr<LegateCompiler>& compiler);

}  // namespace xla

#endif  // HLO_LOADER_H_