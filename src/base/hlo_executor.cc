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
#include <core/data/scalar.h>
#include <type_traits>

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
  BufferAllocation operator()(legate::PhysicalStore &store) {
    using VAL = legate::type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    void *buffer = nullptr;
    auto acc = store.write_accessor<VAL, DIM>();
    buffer = static_cast<void *>(acc.ptr(shape));
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

  log_xla.info() << "HLOExecutorTask: start " << exe->Name();

  auto *callbacks = context.scalars()[ScalarCallbacks]
                        .value<std::vector<std::function<void()>> *>();

  run_executable(context, exe.get(), compiler, run_id, ScalarNumScalarArgs,
                 cpu);
  log_xla.info() << "HLOExecutorTask: finish " << exe->Name();

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
}

template <class Variant, int Index = 0>
BufferAllocation GetScalarVariant(void *buffer, const Scalar &scalar,
                                  uint64_t variant_index) {
  if constexpr (Index == std::variant_size_v<Variant>) {
    throw std::runtime_error(
        "received bad variant index to HloExecutorTask::run_executable");
  } else {
    if (variant_index == Index) {
      // using Value = int32_t;
      using Value = std::decay_t<decltype(std::get<Index>(Variant{}))>;
      auto *sbuffer = static_cast<Value *>(buffer);
      *static_cast<Value *>(buffer) = scalar.value<Value>();
      return {buffer, sizeof(Value)};
    }
    return GetScalarVariant<Variant, Index + 1>(buffer, scalar, variant_index);
  }
}

/*static*/ void HLOExecutorTask::run_executable(legate::TaskContext context,
                                                LegateExecutable *exe,
                                                LegateCompiler *compiler,
                                                int64_t run_id,
                                                int scalar_offset, bool cpu) {
  auto cfg = get_task_config(context);
  log_xla.debug() << "Running task " << exe->Name() << " for device "
                  << cfg.my_device_id << " in range ["
                  << cfg.device_id_range.low << "," << cfg.device_id_range.high
                  << ")"
                  << " with replicas=" << exe->ReplicaCount()
                  << "  and partitions=" << exe->NumPartitions()
                  << ", run_id=" << run_id;

  std::vector<legate_xla::BufferAllocation> inputs, outputs;

  int64_t num_scalar_arguments =
      context.scalar(ScalarNumScalarArgs).value<int64_t>();
  if (num_scalar_arguments > kMaxScalarArguments) {
    throw std::runtime_error(
        "too many scalar arguments passed to HLOExecutorTask");
  }

  using max_size_scalar_t = int64_t;

  std::unordered_map<int64_t, BufferAllocation> scalars;
  DeferredBufferAllocator allocator;
  max_size_scalar_t host_scalar_arguments[kMaxScalarArguments];
  max_size_scalar_t *device_scalar_buffer = host_scalar_arguments;
  static constexpr int kAlignedSegmentSize = 4096;
  size_t num_scalar_segments =
      (num_scalar_arguments * sizeof(max_size_scalar_t) + kAlignedSegmentSize -
       1) /
      kAlignedSegmentSize;
  size_t scalars_size = num_scalar_segments * kAlignedSegmentSize;

  if (!cpu && num_scalar_arguments > 0) {
    device_scalar_buffer =
        static_cast<max_size_scalar_t *>(allocator.Allocate(scalars_size));
  }

  int64_t arg_offset = ScalarNumScalarArgs + 1;
  for (auto i = 0; i < num_scalar_arguments; ++i) {
    uint64_t param_number = context.scalar(arg_offset++).value<uint64_t>();
    uint64_t variant_index = context.scalar(arg_offset++).value<uint64_t>();
    BufferAllocation alloc = GetScalarVariant<ScalarArgument::ValueVariant>(
        &host_scalar_arguments[i], context.scalar(arg_offset++), variant_index);
    alloc.buffer = &device_scalar_buffer[i];
    scalars[param_number] = alloc;
  }

  int input_store_index = 0;
  inputs.reserve(num_scalar_arguments + context.num_inputs());
  for (int param_number = 0;
       param_number < context.num_inputs() + num_scalar_arguments;
       ++param_number) {
    auto iter = scalars.find(param_number);
    if (iter == scalars.end()) {
      auto &&store = context.input(input_store_index++).data();
      inputs.push_back(legate::double_dispatch(
          store.dim(), store.code(), get_read_only_buffer_fn{}, store));
    } else {
      inputs.push_back(iter->second);
    }
  }

  outputs.reserve(context.num_outputs());
  for (const auto &output : context.outputs()) {
    auto &&store = output.data();
    outputs.push_back(legate::double_dispatch(
        store.dim(), store.code(), get_write_only_buffer_fn{}, store));
  }

  if (cfg.num_tasks != (exe->ReplicaCount() * exe->NumPartitions())) {
    std::cerr << exe->Name() << " launched with " << cfg.num_tasks
              << " tasks, but requested device assignment of size "
              << exe->ReplicaCount() << "x" << exe->NumPartitions()
              << std::endl;
    abort();
  }

  if (!cpu && num_scalar_arguments > 0) {
    exe->MemcpyHtoDAsync(device_scalar_buffer, host_scalar_arguments,
                         num_scalar_arguments * sizeof(max_size_scalar_t),
                         cfg.local_device_id);
  }

  DeviceAssignment device_assignment({.local_device_id = cfg.local_device_id,
                                      .global_device_id = cfg.my_device_id,
                                      .replica_count = exe->ReplicaCount(),
                                      .num_partitions = exe->NumPartitions()});
  uint32_t device_id = cfg.device_id_range.low;
  for (int r = 0; r < exe->ReplicaCount(); ++r) {
    for (int c = 0; c < exe->NumPartitions(); ++c) {
      device_assignment(r, c) = device_id++;
    }
  }

  LegateExecutable::Platform platform =
      cpu ? LegateExecutable::CPU : LegateExecutable::GPU;

  static bool blocking = BlockingExecution();

  auto start_clock = std::chrono::steady_clock::now();
  auto error_message = exe->Execute(run_id, inputs, outputs, &allocator,
                                    device_assignment, platform, blocking);

  if (error_message.has_value()) {
    std::cerr << *error_message << std::endl;
    throw std::runtime_error(*error_message);
  }

  if (blocking) {
    auto stop_clock = std::chrono::steady_clock::now();
    auto time_to_execute =
        std::chrono::duration_cast<std::chrono::microseconds>(stop_clock -
                                                              start_clock);
    log_xla.info() << "Ran HLO " << exe->Name() << " in "
                   << (time_to_execute.count() / 1e3) << "ms";
  }

  if (!cpu && num_scalar_arguments > 0) {
    allocator.Free(device_scalar_buffer, scalars_size);
  }

  log_xla.debug() << "Finish task " << exe->Name() << " for device "
                  << cfg.my_device_id;
}

/*static*/ void HLOExecutorTask::cpu_variant(TaskContext context) {
  run_executable(context, /*cpu=*/true);
}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  legate::VariantOptions options;
  options.return_size = 16384;
  options.concurrent = true;

  HLOExecutorTask::register_variants(
      {{LEGATE_CPU_VARIANT, options}, {LEGATE_GPU_VARIANT, options}});
}
} // namespace

} // namespace legate_xla
