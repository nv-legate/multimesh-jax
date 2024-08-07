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
#include "task_utils.h"
#include "xla_task.h"
#include <core/utilities/typedefs.h>

namespace legate_xla {

class HLOExecutorTask : public XlaTask<HLOExecutorTask> {
public:
  static constexpr auto TASK_ID = legate::LocalTaskID{XLA_EXECUTE_TASK};

  enum {
    ScalarTaskCounter = 0,
    ScalarEnforceOrdering,
    ScalarCompilerPointer,
    ScalarRunId,
    ScalarCallbacks,
    ScalarNumScalarArgs,
  };

public:
  static void run_executable(legate::TaskContext context, bool cpu);

  static void run_executable(legate::TaskContext context, const TaskConfig &cfg,
                             LegateExecutable *exe, LegateCompiler *compiler,
                             int64_t run_id, int scalar_offset, bool cpu);

public:
  static void cpu_variant(legate::TaskContext context);

  static void gpu_variant(legate::TaskContext context);
};

} // namespace legate_xla
