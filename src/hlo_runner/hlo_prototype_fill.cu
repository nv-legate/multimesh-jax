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

#include "core/cuda/stream_pool.h"
#include "hlo_prototype_fill.h"

namespace legate_xla {

using namespace Legion;
using namespace legate;

namespace {

template <class VAL> __global__ void fill(VAL *buffer, size_t size) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (; i < size; i += stride) {
    buffer[i] = 0.06 * (i % 16);
  }
}

struct fill_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  void operator()(legate::Store &store) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.write_accessor<VAL, DIM>();
    VAL *buffer = acc.ptr(shape);
    size_t size = store.domain().get_volume();
    auto stream = legate::cuda::StreamPool::get_stream_pool().get_stream();
    fill<<<1024, 1024, 0, stream>>>(buffer, size);
  }
};

} // namespace

/*static*/ void HLOFillTask::gpu_variant(TaskContext context) {
  for (auto &array : context.outputs()) {
    auto store = array.data();
    legate::double_dispatch(store.dim(), store.code(), fill_buffer_fn{}, store);
  }
  for (auto &array : context.reductions()) {
    auto store = array.data();
    legate::double_dispatch(store.dim(), store.code(), fill_buffer_fn{}, store);
  }
}

} // namespace legate_xla
