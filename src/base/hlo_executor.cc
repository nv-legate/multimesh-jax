
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

#include "hlo_executor.h"

#include "allocator.h"
#include "legate_to_xla.h"
#include "legate_xla_common.h"
#include "task_utils.h"
#include "xla_task.h"
#include <chrono>
#include <mutex>
#include <type_traits>

namespace legate_xla {

void RunExecutable(int64_t run_id, std::shared_ptr<LegateCompiler> compiler, std::vector<ScalarArgument> scalars,
                   zuku::ro_vector<zuku::ShardedArray> inputs, zuku::rw_vector<zuku::ShardedArray> outputs){
  throw std::runtime_error("RunExecutable: unimplemented");
}

} // namespace legate_xla
