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
#include "task_utils.h"

namespace legate_xla {

/*static*/ void ShardAssembleTask::gpu_variant(legate::TaskContext context) {
  assemble_shard(context, [](void *buffer, const void *shard, size_t size) {
    cudaMemcpy(buffer, shard, size, cudaMemcpyDeviceToDevice);
  });
}

} // namespace legate_xla
