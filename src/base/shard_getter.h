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
#include <core/data/scalar.h>
#include <cstdint>
#include <memory>
#include <optional>

namespace legate_xla {

class ShardGetterTask : public XlaTask<ShardGetterTask> {
public:
  static constexpr int32_t TASK_ID = XLA_SHARD_GETTER_TASK;

  enum ScalarArgs {
    ScalarBufferPointers,
    ScalarNumShards,
    ScalarTaskWaiter,
    ScalarStoreId
  };

public:
  static void cpu_variant(legate::TaskContext context);

  static void gpu_variant(legate::TaskContext context);

public:
  static void get_shard(legate::TaskContext context);
};

} // namespace legate_xla
