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

#include "allocator.h"
#include "core/utilities/dispatch.h"
#include "legate_to_xla.h"
#include "task_utils.h"
#include "xla_task.h"
#include <chrono>

namespace legate_xla {

using namespace Legion;
using namespace legate;

namespace {

struct get_read_only_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>

  BufferAllocation operator()(legate::Store &store) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.read_accessor<VAL, DIM>();
    size_t size =
        sizeof(legate::legate_type_of<TYPE_CODE>) * store.domain().get_volume();
    void *buffer =
        const_cast<void *>(static_cast<const void *>(acc.ptr(shape)));
    return BufferAllocation{.buffer = buffer, .size = size};
  }
};

struct get_write_only_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  BufferAllocation operator()(legate::Store &store, bool is_red) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    void *buffer = nullptr;
    if (is_red) {
      auto acc = store.reduce_accessor<Legion::SumReduction<VAL>, true, DIM>();
      buffer = static_cast<void *>(acc.ptr(shape));
    } else {
      auto acc = store.write_accessor<VAL, DIM>();
      buffer = static_cast<void *>(acc.ptr(shape));
    }
    size_t size =
        sizeof(legate::legate_type_of<TYPE_CODE>) * store.domain().get_volume();
    return BufferAllocation{.buffer = buffer, .size = size};
  }
};

} // namespace

/*static*/ void HLOExecutorTask::run_executable(legate::TaskContext context) {
  int scalar_offset = 0;
  LegateExecutable *exe = reinterpret_cast<LegateExecutable *>(
      context.scalars()[scalar_offset++].value<void *>());
  uint64_t run_id = context.scalars()[scalar_offset++].value<int64_t>();

  log_xla.debug() << "HLOExecutorTask start";

  auto *callbacks = context.scalars()[scalar_offset++]
                        .value<std::vector<std::function<void()>> *>();

  run_executable(context, exe, run_id, scalar_offset);
  log_xla.debug() << "HLOExecutorTask run_executable done";

  if (false) { // callbacks->size() > 0) {
    log_xla.debug() << "Running total of " << callbacks->size() << " callbacks";
    for (auto &fn : *callbacks) {
      try {
        fn();
      } catch (const std::exception &e) {
        log_xla.error()
            << "Standard exception caught during callback excecution, message '"
            << e.what() << "'";
      } catch (...) {
        log_xla.error() << "Exception caught during callback excecution";
      }
    }
  }
  // delete callbacks;

  log_xla.debug() << "HLOExecutorTask callbacks done";
}

/*static*/ void HLOExecutorTask::run_executable(legate::TaskContext context,
                                                LegateExecutable *exe,
                                                int64_t run_id,
                                                int scalar_offset) {
  std::vector<legate_xla::BufferAllocation> inputs, outputs;

  for (auto &array : context.inputs()) {
    auto store = array.data();
    inputs.push_back(legate::double_dispatch(store.dim(), store.code(),
                                             get_read_only_buffer_fn{}, store));
  }

  size_t total_outputs = context.outputs().size() + context.reductions().size();
  int output_idx = 0;
  int red_idx = 0;

  // first 2 scalars are exe and ID values
  for (size_t idx = scalar_offset; idx < total_outputs + scalar_offset; ++idx) {
    bool is_red = context.scalars()[idx].value<bool>();
    Store store = is_red ? context.reductions()[red_idx++].data()
                         : context.outputs()[output_idx++].data();

    outputs.push_back(legate::double_dispatch(
        store.dim(), store.code(), get_write_only_buffer_fn{}, store, is_red));
  }

  auto cfg = get_task_config(context);

  DeferredBufferAllocator allocator;
  DeviceAssignment device_assignment({.local_device_id = cfg.local_proc_id,
                                      .replica_count = 1,
                                      .num_partitions = cfg.num_tasks});

  for (uint32_t device_id = cfg.device_id_range.low, idx = 0;
       device_id < cfg.device_id_range.high; ++device_id, ++idx) {
    device_assignment(0, idx) = device_id;
  }

  // If this is set to false, the execution profile and ComputeTimeNs for the
  // stream will be incorrect since it will not include the blocking time.
  bool block_host_until_done = true;

  bool success;

  try {
    success =
        exe->Execute(run_id, inputs, outputs, &allocator, device_assignment);
  } catch (const std::exception &e) {
    log_xla.error() << "Standard exception caught during 'Execute', message '"
                    << e.what() << "'";
  } catch (...) {
    log_xla.error() << "Exception caught during 'Execute'";
  }

  // Check that the stream ran and finished correctly
  if (!success) {
    log_xla.error() << "[HLOExecutor] HLO failed!";
#ifdef LEGATE_XLA_PYTHON_PROTOTYPE
    LEGATE_ABORT;
#endif
  }
}

/*static*/ void HLOExecutorTask::cpu_variant(TaskContext context) {
  run_executable(context);
}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  legate::VariantOptions options;
  options.return_size = 16384;

  HLOExecutorTask::register_variants(
      {{LEGATE_CPU_VARIANT, options}, {LEGATE_GPU_VARIANT, options}});
}
} // namespace

} // namespace legate_xla
