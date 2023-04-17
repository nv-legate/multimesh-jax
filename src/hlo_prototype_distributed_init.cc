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

#include "hlo_prototype_distributed_init.h"
#include "legate_to_xla.h"
#include <legate_defines.h>
#include <unistd.h>
#include <chrono>

#include "core/utilities/dispatch.h"

namespace legate_xla {

using namespace Legion;
using namespace legate;

/*static*/ void HloPrototypeDistributedInitTask::init_distributed(legate::TaskContext& context)
{
  std::string coordinator_address = context.scalars()[0].value<std::string>();
  int port                        = context.scalars()[1].value<int32_t>();

  auto device_id_range = context.machine_desc().processor_range();
  auto task_id         = static_cast<int32_t>(context.get_task_index()[0]);
  auto num_tasks       = device_id_range.count();
  auto local_proc_id =
    static_cast<int32_t>((device_id_range.lo + task_id) % device_id_range.per_node_count);
  auto local_node_id =
    static_cast<int32_t>((device_id_range.lo + task_id) / device_id_range.per_node_count);
  auto num_nodes = static_cast<int32_t>((device_id_range.hi - device_id_range.lo + 1) /
                                        device_id_range.per_node_count);

  char my_hostname[1024];
  gethostname(my_hostname, 1024);
  fprintf(stderr, "Init distributed on %s -> %d\n", my_hostname, local_node_id);
  fflush(stderr);
  // only initialize one task per process/node
  if (local_proc_id == 0) {
    bool success = InitDistributedRuntime(
      coordinator_address, port, num_nodes, local_node_id, device_id_range.per_node_count);
    if (!success) { LEGATE_ABORT; }
  }
}

/*static*/ void HloPrototypeDistributedInitTask::cpu_variant(TaskContext& context)
{
  init_distributed(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  HloPrototypeDistributedInitTask::register_variants();
}
}  // namespace

}  // namespace legate_xla
