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

#include "hlo_prototype_fill.h"

using namespace Legion;
using namespace legate;

namespace legate_xla {

namespace {

struct fill_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  void operator()(legate::Store &store) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.write_accessor<VAL, DIM>();
    VAL *buffer = acc.ptr(shape);
    size_t size = store.domain().get_volume();
    for (size_t i = 0; i < size; i++) {
      buffer[i] = 0.06 * (i % 16);
    }
  }
};

} // namespace

/*static*/ void HLOFillTask::cpu_variant(TaskContext &context) {
  for (auto &store : context.outputs()) {
    legate::double_dispatch(store.dim(), store.code(), fill_buffer_fn{}, store);
  }
}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  HLOFillTask::register_variants();
}
} // namespace

} // namespace legate_xla
