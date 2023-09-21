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

#include "shard_getter.h"
#include "allocator.h"
#include "executable_cache.h"
#include "legate_to_xla.h"
#include "shard_getter.h"
#include "task_utils.h"
#include <condition_variable>
#include <mutex>

using namespace legate;

namespace legate_xla {

struct get_read_only_ptr {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  const void *operator()(const legate::Store &store) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.read_accessor<VAL, DIM>();
    const void *buffer = static_cast<const void *>(acc.ptr(shape));
    return buffer;
  }
};

/*static*/ void ShardGetterTask::get_shard(TaskContext context) {
  log_xla.debug() << "ShardGetterTask start";
  auto cfg = get_task_config(context);
  const void **shard_buffers =
      reinterpret_cast<const void **>(context.scalar(0).value<uint64_t>());
  const auto &array = context.input(0);
  const void *buffer = legate::double_dispatch(
      array.dim(), array.data().code(), get_read_only_ptr{}, array.data());

  auto *waiter =
      reinterpret_cast<TaskWaiter *>(context.scalar(1).value<uint64_t>());

  shard_buffers[cfg.local_proc_id] = buffer;
  waiter->Signal();
}

/*static*/ void ShardGetterTask::cpu_variant(TaskContext context) {
  get_shard(context);
}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  ShardGetterTask::register_variants();
}
} // namespace

} // namespace legate_xla
