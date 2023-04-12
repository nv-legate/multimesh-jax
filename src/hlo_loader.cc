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

#include "hlo_loader.h"
#include "allocator.h"
#include "task_utils.h"
#include "legate_xla.h"

using namespace legate;

namespace legate_xla {

/*static*/ void HLOLoaderTask::load_and_compile(TaskContext& context,
                                                const std::string& platform_name)
{
  auto& scalars           = context.scalars();
  LegateCompiler* compiler = reinterpret_cast<LegateCompiler*>(scalars[0].value<void*>());
  uint64_t run_id = scalars[1].value<uint64_t>();
  uint32_t num_partitions = scalars[2].value<uint32_t>();

  auto cfg = get_task_config(context);
  DeferredBufferAllocator allocator;
  compiler->Compile(
    run_id,
    {.replica_count         = 1,  // TODO: we do not handle psum calls or replica all-reduces
     .num_partitions        = (int)num_partitions,
     .run_hlo_passes        = true,
     .stream_executor_index = cfg.local_proc_id,
     .allocator             = &allocator});
}

/*static*/ void HLOLoaderTask::cpu_variant(TaskContext& context)
{
  load_and_compile(context, "cpu");
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  HLOLoaderTask::register_variants();
}
}  // namespace

}  // namespace llm
