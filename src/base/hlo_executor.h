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

namespace legate_xla {

class HLOExecutorTask : public XlaTask<HLOExecutorTask> {
public:
  static const int TASK_ID = XLA_EXECUTE_TASK;

public:
  static void run_executable(legate::TaskContext context);

  static void run_executable(legate::TaskContext context, LegateExecutable *exe,
                             int64_t run_id, int scalar_offset);

public:
  static void cpu_variant(legate::TaskContext context);

  static void gpu_variant(legate::TaskContext context);
};

} // namespace legate_xla
