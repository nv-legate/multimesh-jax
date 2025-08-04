/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mm_utils.h"

#include "absl/strings/str_cat.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/util.h"

namespace xla {
namespace {

namespace se = stream_executor;

struct StreamCacheEntry {
  bool initialized{false};
  se::StreamExecutor* executor_{nullptr};
  std::unique_ptr<se::Stream> stream_{nullptr};

  absl::Status initialize(se::StreamExecutor* executor) {
    if (initialized) return absl::OkStatus();

    initialized = true;
    TF_ASSIGN_OR_RETURN(stream_, executor->CreateStream());
    executor_ = executor;
    return absl::OkStatus();
  }

  void clear() {
    initialized = false;
    executor_ = nullptr;
    stream_ = nullptr;
  }

  se::Stream* stream() const { return stream_.get(); }
};

}  // namespace

static constexpr int MAX_LOCAL_GPUS = 32;
static StreamCacheEntry stream_cache[MAX_LOCAL_GPUS];

void ClearCachedStreams() {
  for (int i = 0; i < MAX_LOCAL_GPUS; ++i) {
    stream_cache[i].clear();
  }
}

absl::StatusOr<se::Stream*> GetCachedStream(se::StreamExecutor* se,
                                            int32_t device_id) {
  if (device_id >= MAX_LOCAL_GPUS) {
    return InvalidArgumentStrCat("Request device=", device_id,
                                 " but stream cache is limited to size ",
                                 MAX_LOCAL_GPUS);
  }
  if (!stream_cache[device_id].initialized) {
    TF_RETURN_IF_ERROR(stream_cache[device_id].initialize(se));
  }
  return stream_cache[device_id].stream();
}

absl::StatusOr<se::Stream*> GetCachedStream(Backend* backend,
                                            int32_t device_id) {
  TF_ASSIGN_OR_RETURN(auto* se, backend->stream_executor(device_id));
  return GetCachedStream(se, device_id);
}

TaskDeviceMemoryAllocator::TaskDeviceMemoryAllocator(
    int device_ordinal, TaskMemoryAllocator* allocator, se::Stream* stream,
    Backend* backend)
    : se::DeviceMemoryAllocator(backend->platform()),
      allocator_(allocator),
      device_ordinal_(device_ordinal),
      stream_(stream) {}

absl::StatusOr<se::OwningDeviceMemory> TaskDeviceMemoryAllocator::Allocate(
    int device_ordinal, uint64_t size, bool retry_on_failure,
    int64_t memory_space) {
  void* buf = allocator_->Allocate(size);
  if (buf == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("allocation of size ", size, " failed"));
  }
  se::DeviceMemoryBase mem(buf, size);
  return se::OwningDeviceMemory(std::move(mem), 0, this);
}

absl::Status TaskDeviceMemoryAllocator::Deallocate(int device_ordinal,
                                                   se::DeviceMemoryBase mem) {
  allocator_->Free(mem.opaque(), mem.size());
  return absl::OkStatus();
}

// Returns a stream pointer on which it is always safe to access memory
// allocated by this allocator. It is not necessary to use the returned stream
// though, as clients may have additional information letting them safely use
// a different stream.
absl::StatusOr<se::Stream*> TaskDeviceMemoryAllocator::GetStream(
    int device_ordinal) {
  if (device_ordinal != device_ordinal_) {
    return InvalidArgument("allocating memory for device not assigned to task");
  }
  return stream_;
}

StreamWrapper::StreamWrapper(PjRtStreamExecutorClient* client, uint64_t run_id,
                             int device_ordinal, se::Stream* stream,
                             xla::DeviceAssignment device_assignment,
                             xla::Backend* backend,
                             TaskMemoryAllocator* allocator)
    : stream_(stream),
      backend_(backend),
      device_assignment_(std::move(device_assignment)) {
  if (allocator) {
    allocator_.emplace(device_ordinal, allocator, stream_, backend_);
  }

  ExecutableRunOptions run_options;
  run_options.set_run_id(RunId(run_id));
  run_options.set_stream(XlaStream());
  run_options.set_device_ordinal(device_ordinal);
  run_options.set_device_assignment(DeviceAssignment());
  run_options.set_allocator(MemoryAllocator());

  service_run_options_ = ServiceExecutableRunOptions(
      run_options, Backend()->StreamBorrowerWithPriority());
  if (client) {
    service_run_options_.mutable_run_options()->set_gpu_executable_run_options(
        client->gpu_run_options());
  }
}

se::DeviceMemoryAllocator* StreamWrapper::MemoryAllocator() {
  if (allocator_.has_value()) {
    return &allocator_.value();
  } else {
    return backend_->memory_allocator();
  }
}

}  // namespace xla
