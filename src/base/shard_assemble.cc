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

#include "shard_assemble.h"
#include "allocator.h"
#include "executable_cache.h"
#include "legate_to_xla.h"
#include "task_utils.h"
#include <condition_variable>
#include <core/data/logical_store.h>
#include <core/data/physical_store.h>
#include <mutex>

using namespace legate;

namespace legate_xla {

struct get_write_ptr {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  auto operator()(const legate::PhysicalStore &store) {
    using VAL = legate::type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.write_accessor<VAL, DIM>();
    void *buffer = static_cast<void *>(acc.ptr(shape));
    return std::make_pair(buffer, store.domain().get_volume() * sizeof(VAL));
  }
};

/*static*/ void ShardAssembleTask::assemble_shard(TaskContext context,
                                                  bool cpu) {
  auto cfg = get_task_config(context);

  const auto &array = context.output(0);

  int64_t num_shards = context.scalar(ScalarNumShards).value<int64_t>();

  log_xla.debug() << "ShardAssembleTask " << cfg.task_id << " start for "
                  << num_shards << " shards";

  auto *stream_hold = reinterpret_cast<TaskArgHold<LegateStream> *>(
      context.scalars()[ScalarCompilerPointer].value<uint64_t>());
  auto *stream = stream_hold->get();

  auto *waiter = reinterpret_cast<TaskFuture *>(
      context.scalar(ScalarTaskWaiter).value<uint64_t>());

  if (cfg.local_device_id < num_shards) {
    void *shard = context.scalar(ScalarBufferPointers + cfg.local_device_id)
                      .value<void *>();
    auto [buffer, size] = legate::double_dispatch(
        array.dim(), array.data().code(), get_write_ptr{}, array.data());
    stream->MemcpyDtoDAsync(buffer, shard, size, cpu, cfg.local_device_id);
    if (waiter){
      waiter->Signal();
    }
  }

  Release(stream_hold, context.machine().processor_range().per_node_count);
}

/*static*/ void ShardAssembleTask::cpu_variant(TaskContext context) {
  assemble_shard(context, /*cpu=*/true);
}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  ShardAssembleTask::register_variants();
}
} // namespace

} // namespace legate_xla
