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

#include "hlo_prototype_distributed_shutdown.h"
#include "legate_to_xla.h"

#include <chrono>

#include "core/utilities/dispatch.h"

namespace legate_xla {

using namespace Legion;
using namespace legate;

/*static*/ void HloPrototypeDistributedShutdownTask::shutdown_distributed(legate::TaskContext& context)
{
  auto device_id_range = context.machine_desc().processor_range();
  auto task_id         = static_cast<int32_t>(context.get_task_index()[0]);
  auto num_tasks       = device_id_range.count();
  auto local_proc_id =
    static_cast<int32_t>((device_id_range.lo + task_id) % device_id_range.per_node_count);

  // only initialize one task per process/node
  if (local_proc_id == 0) {
    log_xla.debug() << "[DistributedShutdownTask] start shutdown on " << task_id;
    ShutdownDistributedRuntime();
    log_xla.debug() << "[DistributedShutdownTask] finish shutdown on " << task_id;
  }
}

/*static*/ void HloPrototypeDistributedShutdownTask::cpu_variant(TaskContext& context)
{
  shutdown_distributed(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  HloPrototypeDistributedShutdownTask::register_variants();
}
}  // namespace

}  // namespace llm
