/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "hlo_executor.h"

#include <chrono>

#include "allocator.h"
#include "src/zuku/processor.h"
#include "src/zuku/profile.h"
#include "src/zuku/type_traits.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {
namespace {

constexpr int64_t kMaxScalarArguments = 64;

class TempBufferAllocator : public TaskMemoryAllocator {
 public:
  explicit TempBufferAllocator(const zuku::ArrayTile& array);

  void* Allocate(size_t size) override;
  void Free(void* buf, size_t size) override;

 private:
  const char* base_ptr_;
  const char* next_ptr_;
  const int64_t size_;
  int64_t freed_size_;
};

TempBufferAllocator::TempBufferAllocator(const zuku::ArrayTile& array)
    : size_(array.byte_size()),
      base_ptr_(array.byte_size() > 0 ? array.ptr<char>() : nullptr),
      freed_size_(0) {
  next_ptr_ = base_ptr_;
}

void* TempBufferAllocator::Allocate(size_t size) {
  const int64_t size_to_allocate = AlignTempSize(size);

  // log_xla.debug() << "Allocating temp of size " << size << " at pointer "
  //                << (void *)next_ptr_;

  void* ret_ptr = const_cast<char*>(next_ptr_);
  next_ptr_ += size_to_allocate;
  if (next_ptr_ > (base_ptr_ + size_)) {
    std::cerr << "requested size " << size
              << " exceeds capacity of temp buffer of size " << size_
              << std::endl;
    abort();
  }

  return ret_ptr;
}

void TempBufferAllocator::Free(void* buf, size_t size) {
  const int64_t size_allocated = AlignTempSize(size);
  freed_size_ += size_allocated;
  // we are a dumb allocator, only reset the next pointer
  // once all previous temp allocations have been freed
  if (freed_size_ == size_) {
    next_ptr_ = base_ptr_;
  }
}

}  // namespace

void RunExecutable(zuku::Stream* zs, int64_t run_id, zuku::DeviceList devices,
                   zuku::Processor p,
                   std::shared_ptr<MultiMeshCompiler> compiler,
                   std::vector<ScalarArgument> scalars,
                   zuku::ro_vector<zuku::ShardedArray> inputs,
                   zuku::rw_vector<zuku::ShardedArray> outputs,
                   const zuku::ArrayTile& temp) {
  // log_xla.debug() << "Executing " << compiler->Name() << " on "
  //                << p.global_id();

  TempBufferAllocator allocator{temp};

  using max_size_scalar_t = int64_t;
  max_size_scalar_t host_scalar_arguments[kMaxScalarArguments];
  max_size_scalar_t* device_scalar_buffer = [&]() {
    if (scalars.empty()) {
      return (max_size_scalar_t*)nullptr;
    }
    if (p.type() == zuku::Processor::Type::GPU) {
      return static_cast<max_size_scalar_t*>(
          allocator.Allocate(sizeof(max_size_scalar_t) * scalars.size()));
    }
    return host_scalar_arguments;
  }();

  std::unordered_map<int64_t, se::DeviceMemoryBase> scalar_buffers;
  // put all the scalar value into a common host buffer
  for (size_t i = 0; i < scalars.size(); ++i) {
    scalar_buffers[scalars[i].parameter_number] = std::visit(
        zuku::overloaded{[&](auto value) {
          auto* host_value_buffer =
              reinterpret_cast<decltype(value)*>(&host_scalar_arguments[i]);
          *host_value_buffer = value;
          return se::DeviceMemoryBase{&device_scalar_buffer[i], sizeof(value)};
        }},
        scalars[i].value);
  }

  std::vector<se::DeviceMemoryBase> input_buffers;
  const int64_t total_num_parameters = scalars.size() + inputs.size();
  input_buffers.reserve(total_num_parameters);
  int64_t input_store_index = 0;
  for (int64_t param_number = 0; param_number < total_num_parameters;
       ++param_number) {
    auto iter = scalar_buffers.find(param_number);
    if (iter != scalar_buffers.end()) {
      input_buffers.push_back(iter->second);
    } else {
      const zuku::ShardedArray& input_array = inputs[input_store_index++];
      if (!input_array.HasTile()) {
        std::cerr << "No tile on " << p.global_id() << " for array "
                  << input_array.shape() << std::endl;
        abort();
      }
      // XLA requires void* for inputs - so just cast
      input_buffers.emplace_back(const_cast<void*>(input_array.tile().data()),
                                 input_array.tile().byte_size());
    }
  }

  std::vector<se::DeviceMemoryBase> output_buffers;
  output_buffers.reserve(outputs.size());
  for (auto&& output : outputs) {
    output_buffers.emplace_back(output.tile().data(),
                                output.tile().byte_size());
  }

  auto exe = compiler->MakeExecutable();
  if (p.type() == zuku::Processor::Type::GPU) {
    // the scalar values must be transferred to the device if GPU
    exe->MemcpyHtoDAsync(device_scalar_buffer, host_scalar_arguments,
                         scalars.size() * sizeof(max_size_scalar_t),
                         p.local_id());
  }

  const int64_t num_local_devices = p.NumLocalInDeviceList(devices, p.type());
  MultiMeshDeviceAssignment device_assignment{
      {.local_device_id = p.local_id(),
       .global_device_id = p.global_id(),
       .replica_count = exe->ReplicaCount(),
       .num_partitions = exe->NumPartitions()},
      std::move(devices)};

  MultiMeshExecutable::Platform platform =
      p.type() == zuku::Processor::Type::GPU ? MultiMeshExecutable::GPU
                                             : MultiMeshExecutable::CPU;

  static bool blocking = false;  // BlockingExecution();

  auto start_clock = std::chrono::steady_clock::now();
  auto error_message =
      exe->Execute(zs, run_id, input_buffers, output_buffers, &allocator,
                   device_assignment, num_local_devices, platform, blocking);

  if (error_message.has_value()) {
    throw std::runtime_error(*error_message);
  }
}

}  // namespace xla
