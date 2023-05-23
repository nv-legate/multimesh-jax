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
#include "legate_to_xla.h"
#include "task_utils.h"

using namespace legate;

namespace legate_xla {

namespace {

struct get_write_only_ptr {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  void *operator()(legate::Store &store) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.write_accessor<VAL, DIM>();
    void *buffer = static_cast<void *>(acc.ptr(shape));
    return buffer;
  }
};

} // namespace

/*static*/ void HLOLoaderTask::load_and_compile(
    TaskContext &context, LegateCompiler *compiler, uint64_t run_id,
    const std::string &platform_name, std::optional<uint32_t> num_partitions,
    bool has_sync_store, bool print_stats) {
  auto cfg = get_task_config(context);
  uint32_t partitions =
      num_partitions.has_value() ? *num_partitions : cfg.num_tasks;
  DeferredBufferAllocator allocator;
  // only print stats (if requested) on the lowest node
  print_stats = print_stats && (cfg.my_node == cfg.min_node);

  compiler->Compile(
      run_id,
      {.replica_count =
           1, // TODO: we do not handle psum calls or replica all-reduces
       .num_partitions = (int)partitions,
       .run_hlo_passes = true,
       .stream_executor_index = cfg.local_proc_id,
       .allocator = &allocator,
       .print_stats = print_stats});

  // FIXME: could this run on multiple gpus?
  if (has_sync_store) {
    auto &sync_store = context.outputs()[0];
    auto output_ptr = legate::double_dispatch(
        sync_store.dim(), sync_store.code(), get_write_only_ptr{}, sync_store);
  }
}

/*static*/ void
HLOLoaderTask::load_and_compile(TaskContext &context,
                                const std::string &platform_name) {
  auto &scalars = context.scalars();
  LegateCompiler *compiler =
      reinterpret_cast<LegateCompiler *>(scalars[0].value<void *>());
  uint64_t run_id = scalars[1].value<uint64_t>();
  load_and_compile(context, compiler, run_id, platform_name);
}

/*static*/ void HLOLoaderTask::cpu_variant(TaskContext &context) {
  load_and_compile(context, "cpu");
}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  HLOLoaderTask::register_variants();
}
} // namespace

} // namespace legate_xla
