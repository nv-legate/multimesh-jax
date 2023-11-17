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
#include "legate_xla_common.h"
#include "task_utils.h"
#include "xla_task.h"
#include <chrono>

namespace legate_xla {

using namespace Legion;
using namespace legate;

namespace {

struct get_read_only_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>

  BufferAllocation operator()(legate::PhysicalStore &store) {
    using VAL = legate::type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.read_accessor<VAL, DIM>();
    size_t size =
        sizeof(legate::type_of<TYPE_CODE>) * store.domain().get_volume();
    void *buffer =
        const_cast<void *>(static_cast<const void *>(acc.ptr(shape)));
    return BufferAllocation{.buffer = buffer, .size = size};
  }
};

struct get_write_only_buffer_fn {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  BufferAllocation operator()(legate::PhysicalStore &store, bool is_red) {
    using VAL = legate::type_of<TYPE_CODE>;
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
        sizeof(legate::type_of<TYPE_CODE>) * store.domain().get_volume();
    return BufferAllocation{.buffer = buffer, .size = size};
  }
};

} // namespace

/*static*/ void HLOExecutorTask::run_executable(legate::TaskContext context,
                                                bool cpu) {
  auto *compiler_hold = reinterpret_cast<TaskArgHold<LegateCompiler> *>(
      context.scalars()[ScalarCompilerPointer].value<void *>());
  auto *compiler = compiler_hold->get();
  auto exe = compiler->MakeExecutable();
  uint64_t run_id = context.scalars()[ScalarRunId].value<int64_t>();

  log_xla.debug() << "HLOExecutorTask start " << exe->Name();

  auto *callbacks = context.scalars()[ScalarCallbacks]
                        .value<std::vector<std::function<void()>> *>();

  run_executable(context, exe.get(), run_id, NumScalarArgs, cpu);
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
  Release(compiler_hold, context.machine().processor_range().per_node_count);

  log_xla.debug() << "HLOExecutorTask callbacks done";
}

/*static*/ void HLOExecutorTask::run_executable(legate::TaskContext context,
                                                LegateExecutable *exe,
                                                int64_t run_id,
                                                int scalar_offset, bool cpu) {
  auto cfg = get_task_config(context);
  log_xla.debug() << "Running task " << exe->Name() << " for device "
                  << cfg.my_device_id << " in range ["
                  << cfg.device_id_range.low << "," << cfg.device_id_range.high
                  << ")"
                  << " with replicas=" << exe->ReplicaCount()
                  << "  and partitions=" << exe->NumPartitions();
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
    auto store = is_red ? context.reductions()[red_idx++].data()
                        : context.outputs()[output_idx++].data();

    outputs.push_back(legate::double_dispatch(
        store.dim(), store.code(), get_write_only_buffer_fn{}, store, is_red));
  }

  if (cfg.num_tasks != (exe->ReplicaCount() * exe->NumPartitions())) {
    std::cerr << exe->Name() << " launched with " << cfg.num_tasks
              << " tasks, but requested device assignment of size "
              << exe->ReplicaCount() << "x" << exe->NumPartitions()
              << std::endl;
    abort();
  }

  DeferredBufferAllocator allocator;
  DeviceAssignment device_assignment({.local_device_id = cfg.local_device_id,
                                      .replica_count = exe->ReplicaCount(),
                                      .num_partitions = exe->NumPartitions()});
  uint32_t device_id = cfg.device_id_range.low;
  for (int r = 0; r < exe->ReplicaCount(); ++r) {
    for (int c = 0; c < exe->NumPartitions(); ++c) {
      device_assignment(r, c) = device_id++;
    }
  }

  bool success;
  try {
    success = exe->Execute(run_id, inputs, outputs, &allocator,
                           device_assignment, cpu);
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
  run_executable(context, /*cpu=*/true);
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
