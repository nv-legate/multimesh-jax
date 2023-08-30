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

class HLOPrototypeLoaderTask : public XlaTask<HLOPrototypeLoaderTask> {
public:
  static constexpr int32_t TASK_ID = HLO_PROTOTYPE_LOAD;

public:
  static void cpu_variant(legate::TaskContext context);

  static void gpu_variant(legate::TaskContext context);

public:
  static void load_and_compile(legate::TaskContext context,
                               const std::string &platform_name);
};

} // namespace legate_xla
