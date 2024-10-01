
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
#include "legate_to_xla.h"
#include "legate_xla_common.h"
#include "task_utils.h"
#include "xla_task.h"
#include <chrono>
#include <mutex>
#include <processor.h>
#include <type_traits.h>
#include <type_traits>

namespace legate_xla {
namespace {

constexpr int64_t kMaxScalarArguments = 64;

}

void RunExecutable(int64_t run_id, zuku::DeviceList devices, zuku::Processor p,
                   std::shared_ptr<LegateCompiler> compiler,
                   std::vector<ScalarArgument> scalars,
                   zuku::ro_vector<zuku::ShardedArray> inputs,
                   zuku::rw_vector<zuku::ShardedArray> outputs,
                   const zuku::ArrayTile &temp) {
  TempBufferAllocator allocator{temp};

  using max_size_scalar_t = int64_t;
  max_size_scalar_t host_scalar_arguments[kMaxScalarArguments];
  max_size_scalar_t *device_scalar_buffer = [&] {
    if (p.type() == zuku::Processor::Type::GPU) {
      return static_cast<max_size_scalar_t *>(
          allocator.Allocate(sizeof(max_size_scalar_t) * scalars.size()));
    }
    return host_scalar_arguments;
  }();

  std::unordered_map<int64_t, BufferAllocation> scalar_buffers;
  // put all the scalar value into a common host buffer
  for (size_t i = 0; i < scalars.size(); ++i) {
    scalar_buffers[scalars[i].parameter_number] = std::visit(
        zuku::overloaded{[&](auto value) {
          using value_t = decltype(value);
          auto *host_value_buffer =
              reinterpret_cast<decltype(value) *>(&host_scalar_arguments[i]);
          *host_value_buffer = value;
          return BufferAllocation{.buffer = &device_scalar_buffer[i],
                                  .size = sizeof(value)};
        }},
        scalars[i].value);
  }

  std::vector<legate_xla::BufferAllocation> input_buffers;
  const int64_t total_num_parameters = scalars.size() + inputs.size();
  input_buffers.reserve(total_num_parameters);
  int64_t input_store_index = 0;
  for (int64_t param_number = 0; param_number < total_num_parameters;
       ++param_number) {
    auto iter = scalar_buffers.find(param_number);
    if (iter != scalar_buffers.end()) {
      input_buffers.push_back(iter->second);
    } else {
      const zuku::ShardedArray &input_array = inputs[input_store_index++];
      input_buffers.push_back(BufferAllocation{
          .buffer = const_cast<void *>(
              input_array.tile()
                  .data()), // XLA requires void* for inputs - so just cast
          .size = input_array.tile().byte_size()});
    }
  }

  std::vector<legate_xla::BufferAllocation> output_buffers;
  output_buffers.reserve(outputs.size());
  for (auto &&output : outputs) {
    output_buffers.push_back(BufferAllocation{
        .buffer = output.tile().data(),
        .size = output.tile().byte_size(),
    });
  }

  auto exe = compiler->MakeExecutable();
  if (p.type() == zuku::Processor::Type::GPU) {
    // the scalar values must be transferred to the device if GPU
    exe->MemcpyHtoDAsync(device_scalar_buffer, host_scalar_arguments,
                         scalars.size() * sizeof(max_size_scalar_t),
                         p.local_id());
  }

  DeviceConfig device_cfg{.local_device_id = p.local_id(),
                          .global_device_id = p.global_id(),
                          .replica_count = exe->ReplicaCount(),
                          .num_partitions = exe->NumPartitions()};
  DeviceAssignment device_assignment{device_cfg, std::move(devices)};

  LegateExecutable::Platform platform = p.type() == zuku::Processor::Type::GPU
                                            ? LegateExecutable::GPU
                                            : LegateExecutable::CPU;

  static bool blocking = BlockingExecution();

  auto start_clock = std::chrono::steady_clock::now();
  auto error_message =
      exe->Execute(run_id, input_buffers, output_buffers, &allocator,
                   device_assignment, platform, blocking);

  if (error_message.has_value()) {
    std::cerr << *error_message << std::endl;
    throw std::runtime_error(*error_message);
  }
}

} // namespace legate_xla
