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

#include "core/cuda/cuda_help.h"
#include "hlo_prototype_distributed_init.h"

namespace legate_xla {

/*static*/ void
HloPrototypeDistributedInitTask::gpu_variant(legate::TaskContext context) {
  init_distributed(context);
  CHECK_CUDA(cudaPeekAtLastError());
}

} // namespace legate_xla
