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
#include "executable_cache.h"
#include "legate_to_xla.h"
#include "task_utils.h"

using namespace legate;

namespace legate_xla {

/*static*/ void HLOLoaderTask::load_and_compile(
    TaskContext context, LegateCompiler *compiler, uint64_t run_id,
    const std::string &platform_name, const HloLoaderOptions &options) {

  uint64_t hlo_id =
      options.hlo_id.has_value() ? *options.hlo_id : compiler->HloId();
  // only one GPU per node should be running the compilation
  compile_executable(hlo_id, [&] {
    auto cfg = get_task_config(context);
    uint32_t partitions = options.num_partitions.has_value()
                              ? *options.num_partitions
                              : cfg.num_tasks;
    DeferredBufferAllocator allocator;
    // only print stats (if requested) on the lowest node
    bool print_stats = options.print_stats && (cfg.my_node == cfg.min_node);

    try {
      compiler->Compile(
          run_id,
          {.replica_count =
               1, // TODO: we do not handle psum calls or replica all-reduces
           .num_partitions = (int)partitions,
           .run_hlo_passes = true,
           .stream_executor_index = cfg.local_device_id,
           .allocator = &allocator,
           .print_stats = print_stats});
    } catch (const std::exception &e) {
      log_xla.error() << "Standard exception caught during 'Compile', message '"
                      << e.what() << "'";
    } catch (...) {
      log_xla.error() << "Exception caught during 'Compile'";
    }
    return compiler->MakeExecutable();
  });
}

/*static*/ void
HLOLoaderTask::load_and_compile(TaskContext context,
                                const std::string &platform_name) {
  log_xla.debug() << "HLOLoaderTask start";
  auto &scalars = context.scalars();
  LegateCompiler *compiler =
      reinterpret_cast<LegateCompiler *>(scalars[0].value<void *>());
  uint64_t run_id = scalars[1].value<uint64_t>();
  load_and_compile(context, compiler, run_id, platform_name);
  log_xla.debug() << "HLOLoaderTask done";
}

/*static*/ void HLOLoaderTask::cpu_variant(TaskContext context) {
  load_and_compile(context, "cpu");
}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  HLOLoaderTask::register_variants();
}
} // namespace

} // namespace legate_xla
