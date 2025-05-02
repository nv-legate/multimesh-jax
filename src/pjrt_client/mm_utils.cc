/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mm_utils.h"

#include "absl/strings/str_cat.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/gpu/nccl_id_store.h"
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

struct DistributedRuntime {
  std::unique_ptr<DistributedRuntimeService> runtime_service;
  std::shared_ptr<DistributedRuntimeClient> runtime_client;
  gpu::GpuExecutableRunOptions gpu_executable_run_options;
};

DistributedRuntime* GetDistributedRuntime() {
  static DistributedRuntime runtime;
  return &runtime;
}

}  // namespace

absl::Status InitDistributedRuntimeParams(
    int num_procs, int node_id, int gpus_per_node,
    std::shared_ptr<KeyValueStoreInterface> kv_store) {
  std::map<int, GlobalDeviceId> gpu_device_ids;
  int local_gpu_start = node_id * gpus_per_node;
  for (int gpu = 0; gpu < gpus_per_node; ++gpu) {
    gpu_device_ids[gpu] = local_gpu_start + gpu;
  }
  GetDistributedRuntime()->gpu_executable_run_options.set_gpu_global_device_ids(
      std::move(gpu_device_ids));

  absl::flat_hash_map<GlobalDeviceId, int> device_to_node;
  for (int node = 0; node < num_procs; ++node) {
    int gpu_start = node * gpus_per_node;
    int gpu_stop = gpu_start + gpus_per_node;
    for (int gpu = gpu_start; gpu < gpu_stop; ++gpu) {
      GlobalDeviceId global_id(gpu);
      device_to_node[global_id] = node;
    }
  }

  if (num_procs > 1) {
    auto nccl_id_store =
        std::make_shared<NcclIdStore>(node_id, device_to_node, kv_store);
    GetDistributedRuntime()->gpu_executable_run_options.set_clique_id_callback(
        [nccl_id_store](const CliqueKey& key) {
          return nccl_id_store->GetNcclUniqueId(key);
        });
  }

  return absl::OkStatus();
}

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
    return tsl::errors::InvalidArgument(
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

StreamWrapper::StreamWrapper(uint64_t run_id, int device_ordinal,
                             se::Stream* stream,
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
  service_run_options_.mutable_run_options()->set_gpu_executable_run_options(
      &GetDistributedRuntime()->gpu_executable_run_options);
}

se::DeviceMemoryAllocator* StreamWrapper::MemoryAllocator() {
  if (allocator_.has_value()) {
    return &allocator_.value();
  } else {
    return backend_->memory_allocator();
  }
}

std::shared_ptr<DistributedRuntimeClient> MultiMeshRuntimeClient() {
  return GetDistributedRuntime()->runtime_client;
}

}  // namespace xla
