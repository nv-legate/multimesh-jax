/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "legate_xla_common.h"
#include "xla_task.h"
#include <zuku/store.h>
#include <cstdint>
#include <memory>
#include <optional>


namespace legate_xla {

struct HloLoaderOptions {
  bool print_stats = false;
  std::optional<uint32_t> num_partitions = std::nullopt;
  std::optional<uint64_t> hlo_id = std::nullopt;
};

void LoadAndCompile(int64_t run_id, const std::shared_ptr<LegateCompiler>& compiler);

} // namespace legate_xla
