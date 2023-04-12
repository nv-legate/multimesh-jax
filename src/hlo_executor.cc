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

#include "hlo_executor.h"

#include <chrono>

#include "legate_xla.h"
#include "xla_task.h"
#include "allocator.h"
#include "task_utils.h"
#include "core/utilities/dispatch.h"

namespace legate_xla {

using namespace Legion;
using namespace legate;

namespace {

struct get_read_only_buffer_fn {
  template <legate::LegateTypeCode TYPE_CODE, int32_t DIM>
  legate::BufferAllocation operator()(legate::Store& store)
  {
    using VAL    = legate::legate_type_of<TYPE_CODE>;
    auto shape   = store.shape<DIM>();
    auto acc     = store.read_accessor<VAL, DIM>();
    size_t size  = sizeof(legate::legate_type_of<TYPE_CODE>) * store.domain().get_volume();
    void* buffer = const_cast<void*>(static_cast<const void*>(acc.ptr(shape)));
    return legate::BufferAllocation{.buffer = buffer, .size = size};
  }
};

struct get_write_only_buffer_fn {
  template <legate::LegateTypeCode TYPE_CODE, int32_t DIM>
  legate::BufferAllocation operator()(legate::Store& store, bool is_red)
  {
    using VAL    = legate::legate_type_of<TYPE_CODE>;
    auto shape   = store.shape<DIM>();
    void* buffer = nullptr;
    if (is_red) {
      auto acc = store.reduce_accessor<Legion::SumReduction<VAL>, true, DIM>();
      buffer   = static_cast<void*>(acc.ptr(shape));
    } else {
      auto acc = store.write_accessor<VAL, DIM>();
      buffer   = static_cast<void*>(acc.ptr(shape));
    }
    size_t size = sizeof(legate::legate_type_of<TYPE_CODE>) * store.domain().get_volume();
    return legate::BufferAllocation{.buffer = buffer, .size = size};
  }
};

}  // namespace

/*static*/ void HLOExecutorTask::run_executable(legate::TaskContext& context)
{
  LegateExecutable* exe = reinterpret_cast<LegateExecutable*>(context.scalars()[0].value<void*>());
  uint64_t run_id = context.scalars()[1].value<uint64_t>();

  std::vector<legate::BufferAllocation> inputs, outputs;

  for (auto& store : context.inputs()) {
    inputs.push_back(
      legate::double_dispatch(store.dim(), store.code(), get_read_only_buffer_fn{}, store));
  }

  size_t total_outputs = context.outputs().size() + context.reductions().size();
  int output_idx       = 0;
  int red_idx          = 0;
  // first 2 scalars are exe and ID values
  int scalar_offset = 2;
  for (size_t idx = scalar_offset; idx < total_outputs + scalar_offset; ++idx) {
    bool is_red  = context.scalars()[idx].value<char>();
    Store& store = is_red ? context.reductions()[red_idx++] : context.outputs()[output_idx++];

    outputs.push_back(legate::double_dispatch(
      store.dim(), store.code(), get_write_only_buffer_fn{}, store, is_red));
  }

  auto cfg = get_task_config(context);

  DeferredBufferAllocator allocator;
  legate::DeviceAssignment device_assignment(
    {.local_device_id = cfg.local_proc_id, .replica_count = 1, .num_partitions = cfg.num_tasks});

  // TODO: need resource scoping to set the device assignment
  device_assignment(0,0) = 0;

  //for (uint32_t device_id = cfg.device_id_range.lo, idx = 0; device_id <= cfg.device_id_range.hi;
  //     ++device_id, ++idx){
  //  device_assignment(0, idx) = device_id;
  //}

  // If this is set to false, the execution profile and ComputeTimeNs for the stream
  // will be incorrect since it will not include the blocking time.
  bool block_host_until_done = true;

  auto ts_start = std::chrono::high_resolution_clock::now();
  auto stream   = exe->Execute(
    run_id, inputs, outputs, &allocator, device_assignment, block_host_until_done);
  // Check that the stream ran and finished correctly
  if (!stream->BlockUntilDone()) {
    log_xla.error() << "[HLOExecutor] HLO failed!";
    LEGATE_ABORT;
  }

}

/*static*/ void HLOExecutorTask::cpu_variant(TaskContext& context) { run_executable(context); }

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void)
{
  legate::VariantOptions options;
  options.return_size = 16384;
  HLOExecutorTask::register_variants(
    {{LEGATE_CPU_VARIANT, options}, {LEGATE_GPU_VARIANT, options}});
}
}  // namespace

}  // namespace llm
