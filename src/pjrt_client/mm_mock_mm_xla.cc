/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mm_mock_mm_xla.h"

#include "absl/hash/hash.h"
#include "xla/pjrt/multimesh/mm_computation.h"

// mock functions that won't be called that are necessary to link test
namespace xla {

namespace {

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  os << "[";
  for (auto v : vec) {
    os << v << ",";
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, MemoryKind kind) {
  if (kind == kHOST) {
    os << "HOST";
  } else if (kind == kDEVICE) {
    os << "DEVICE";
  }
  return os;
}

}  // namespace

struct StoreHandleImpl {
  zuku::ShardedShape shape;
  bool offloaded{false};
  LastOp last_op{LastOp::Created};
  bool has_tile{false};
  std::string name{};
};

// Destructor cannot be defined in the header file due to PIMPL
// NOLINTNEXTLINE(modernize-use-equals-default)
StoreHandle::~StoreHandle() {}

template <typename H>
H AbslHashValue(H h, const StoreHandle& handle) {
  return H::combine(std::move(h), handle.unique_id, handle.impl->name,
                    handle.impl->shape);
}

void MockArrayCache::Free(const zuku::ShardedShape& shape, MemoryKind kind,
                          MockZukuExecuteContext& context) {
  VLOG(5) << "cache freeing " << shape << " for kind=" << kind;
  num_available_++;
  if (num_available_ > num_allocated_) {
    std::stringstream sstr;
    sstr << "mock array cache has more available than allocated for shape "
         << shape << " on " << kind;
    throw std::runtime_error(sstr.str());
  }
}

void MockArrayCache::Allocate(int64_t local_device_id,
                              const zuku::ShardedShape& shape, MemoryKind kind,
                              MockZukuExecuteContext& context) {
  if (num_available_ > 0) {
    VLOG(5) << "cache returning existing " << shape << " for kind=" << kind;
    --num_available_;
  } else {
    VLOG(5) << "cache allocating new " << shape << " for kind=" << kind;
    context.AllocateFromDevice(kind, local_device_id, zuku::ShardSize(shape));
    ++num_allocated_;
  }
}

void MockArrayCache::Clear(int64_t local_device_id, MemoryKind kind,
                           const zuku::ShardedShape& shape,
                           MockZukuExecuteContext& context) {
  context.DeallocateFromDevice(kind, local_device_id,
                               zuku::ShardSize(shape) * num_allocated_);
  num_allocated_ = 0;
  num_available_ = 0;
}

void MockZukuExecuteContext::AllocateFromDevice(MemoryKind kind, int64_t device,
                                                int64_t bytes) {
  auto& pool = device_memories_[device][kind];
  pool.allocated += bytes;
  pool.high_watermark = std::max(pool.allocated, pool.high_watermark);
}

void MockZukuExecuteContext::DeallocateFromDevice(MemoryKind kind,
                                                  int64_t device,
                                                  int64_t bytes) {
  device_memories_[device][kind].allocated -= bytes;
}

void MockZukuExecuteContext::Reset() {
  store_id_counter_ = 0;
  operation_hash_ = 0;
  compute_hashes_ = false;
  compile_modules_ = false;
  created_store_counts_.clear();
  host_caches_.clear();
  device_caches_.clear();
  device_memories_.clear();
}

int64_t MockZukuExecuteContext::HostBytesHighWatermark(
    int64_t local_device_id) {
  return device_memories_[local_device_id][kHOST].high_watermark;
}

int64_t MockZukuExecuteContext::DeviceBytesHighWatermark(
    int64_t local_device_id) {
  return device_memories_[local_device_id][kDEVICE].high_watermark;
}

void MockZukuExecuteContext::Rename(StoreHandle& handle, std::string name) {
  handle.impl->name = std::move(name);
}

void MockZukuExecuteContext::OpenWindow() {}

void MockZukuExecuteContext::CloseWindow() {}

void MockZukuExecuteContext::OffloadHtoD(
    int64_t local_device_id, const std::vector<StoreHandle>& to_offload,
    const std::string& task_name) {
  if (compute_hashes_) {
    operation_hash_ = absl::HashOf(operation_hash_, to_offload, kOffloadHtoD);
  }

  for (auto& handle : to_offload) {
    handle.impl->offloaded = false;
    handle.impl->last_op = LastOp::OffloadHtoD;
    VLOG(3) << "OffloadHtoD: " << handle.impl->name << " on " << local_device_id
            << " " << handle.impl->shape;
    if (device_caches_[local_device_id][handle.impl->shape].Empty()) {
      VLOG(3) << "Allocating new: " << handle.impl->name << " on "
              << local_device_id << " " << handle.impl->shape;
    }
    if (handle.impl->shape.sharding.devices.Contains(local_device_id)) {
      device_caches_[local_device_id][handle.impl->shape].Allocate(
          local_device_id, handle.impl->shape, kDEVICE, *this);
      host_caches_[local_device_id][handle.impl->shape].Free(handle.impl->shape,
                                                             kHOST, *this);
    }
  }
}

void MockZukuExecuteContext::OffloadDtoH(
    int64_t local_device_id, const std::vector<StoreHandle>& to_offload,
    const std::vector<StoreHandle>& pipelined, const std::string& task_name) {
  if (compute_hashes_) {
    operation_hash_ =
        absl::HashOf(operation_hash_, to_offload, pipelined, kOffloadDtoH);
  }

  for (auto& handle : pipelined) {
    // the last op for this handle should be a reshard operation
    // unless it is replicated, in which case the reshard might get skipped
    if (handle.impl->last_op != LastOp::Reshard &&
        !handle.impl->shape.sharding.IsReplicated()) {
      throw std::runtime_error(absl::StrCat(
          handle.impl->name, " has not finished its reshard before offloading ",
          task_name));
    }
  }

  for (auto& handle : to_offload) {
    VLOG(3) << "OffloadDtoH: " << handle.impl->name << " on " << local_device_id
            << " " << handle.impl->shape;
    handle.impl->offloaded = true;
    handle.impl->last_op = LastOp::OffloadDtoH;
    if (handle.impl->shape.sharding.devices.Contains(local_device_id)) {
      device_caches_[local_device_id][handle.impl->shape].Free(
          handle.impl->shape, kDEVICE, *this);
      host_caches_[local_device_id][handle.impl->shape].Allocate(
          local_device_id, handle.impl->shape, kHOST, *this);
    }
  }
}

void MockZukuExecuteContext::ClearStoreCache(int64_t local_device_id) {
  for (auto& [key, cache] : device_caches_[local_device_id]) {
    if (!cache.Full()) {
      throw std::runtime_error(
          "device cache is being cleared, but not all allocations from cache "
          "have been freed");
    }
    cache.Clear(local_device_id, kDEVICE, key, *this);
  }
}

void MockZukuExecuteContext::Free(int64_t local_device_id, StoreHandle handle,
                                  bool keep_in_cache) {
  if (handle.impl->shape.sharding.devices.Contains(local_device_id)) {
    if (keep_in_cache) {
      device_caches_[local_device_id][handle.impl->shape].Free(
          handle.impl->shape, kDEVICE, *this);
    } else {
      DeallocateFromDevice(kDEVICE, local_device_id,
                           zuku::ShardSize(handle.impl->shape));
    }
  }
}

StoreHandle MockZukuExecuteContext::CreateStoreImpl(int64_t local_device_id,
                                                    int64_t global_device_id,
                                                    zuku::ShardedShape shape,
                                                    CreateStoreConfig config) {
  if (compute_hashes_) {
    operation_hash_ = absl::HashOf(kCreateStore, operation_hash_, shape,
                                   local_device_id, global_device_id);
  }

  if (config.name.has_value()) {
    created_store_counts_[*config.name]++;
  }

  if (shape.sharding.devices.Contains(local_device_id)) {
    if (config.allocate_from_cache) {
      device_caches_[local_device_id][shape].Allocate(local_device_id, shape,
                                                      kDEVICE, *this);
    } else {
      AllocateFromDevice(kDEVICE, local_device_id, zuku::ShardSize(shape));
    }
  }

  VLOG(3) << "Create store " << config.name.value_or("anonymous")
          << " with shape=" << shape
          << ", allocate_from_cache=" << config.allocate_from_cache
          << ", min_cache_size=" << config.min_cache_size.value_or(1);

  return {.impl = std::shared_ptr<StoreHandleImpl>{new StoreHandleImpl{
              .shape = shape,
              .has_tile = shape.sharding.devices.Contains(global_device_id),
              .name = config.name.value_or("anonymous")}},
          .unique_id = store_id_counter_++};
}

zuku::Future<zuku::ArrayTile> MockZukuExecuteContext::CreateBuffer(
    int64_t local_device_id, int64_t global_device_id, int64_t size) {
  if (compute_hashes_) {
    operation_hash_ = absl::HashOf(kCreateBuffer, operation_hash_, size);
  }
  return zuku::Future<zuku::ArrayTile>::CreateEmpty();
}

void MockZukuExecuteContext::Reshard(int64_t local_device_id,
                                     int64_t global_device_id,
                                     const StoreHandle& src,
                                     const StoreHandle& dst) {
  if (compute_hashes_) {
    operation_hash_ = absl::HashOf(kReshard, operation_hash_, local_device_id,
                                   global_device_id, src, dst);
  }

  VLOG(3) << "Reshard " << src.impl->name << " from shape " << src.impl->shape
          << " to " << dst.impl->shape;

  src.impl->last_op = LastOp::Reshard;
  dst.impl->last_op = LastOp::Reshard;
}

void MockZukuExecuteContext::Destroy(StoreHandle& store) {}

void MockZukuExecuteContext::MarkProfile(const std::string& name) {}

void MockZukuExecuteContext::StartTimer(const std::string& name) {}

void MockZukuExecuteContext::StopTimer(const std::string& name) {}

void MockZukuExecuteContext::FenceCompilation() {}

zuku::ShardedShape MockZukuExecuteContext::GetStoreShardedShape(
    const StoreHandle& handle) {
  return handle.impl->shape;
}

StoreHandle MockZukuExecuteContext::AssembleShardsImpl(
    int64_t local_device_id, int64_t global_device_id, zuku::ShardedShape shape,
    Shard shard, std::shared_ptr<MultiMeshStream> stream,
    std::optional<StoreHandle> existing_store) {
  if (existing_store.has_value()) {
    return *std::move(existing_store);
  }
  return CreateStore(local_device_id, global_device_id, shape,
                     {.name = "assemble"});
}

std::set<int> MockZukuExecuteContext::GetLocalDevices() { return {}; }

bool MockZukuExecuteContext::IsGpu() {
  // tests are cpu
  return false;
}

void* MockZukuExecuteContext::SliceLocalShard(int64_t local_device_id,
                                              const StoreHandle& handle) {}

bool MockZukuExecuteContext::HasLocalShard(const StoreHandle& handle) {
  return true;
}

void MockZukuExecuteContext::CreateCompileTask(
    int64_t local_device_id, std::shared_ptr<MultiMeshCompiler> compiler) {
  xla::MultiMeshCompiler* xla_compiler =
      dynamic_cast<xla::MultiMeshCompiler*>(compiler.get());
  if (compute_hashes_) {
    operation_hash_ = absl::HashOf(operation_hash_, kCreateCompileTask,
                                   compiler->LaunchSize());

    if (xla_compiler) {
      std::string fingerprint = xla_compiler->module().GetFingerprint128();
      operation_hash_ = absl::HashOf(operation_hash_, fingerprint);
      VLOG(3) << "fingerprint=" << fingerprint;
      name_to_fingerprint_[xla_compiler->Name()] = std::move(fingerprint);
    }
  }
  if (compile_modules_ && local_device_id == 0) {
    MultiMeshCompileConfig config{
        /*replica_count=*/1,
        /*num_partitions=*/(int)compiler->LaunchSize(),
        /*run_hlo_passes=*/true,
        /*run_backend=*/false,
        /*stream_executor_index=*/0};
    xla_compiler->Compile(0, config);
  }
}

void MockZukuExecuteContext::CreateExecuteTask(
    int64_t run_id, int64_t local_device_id, int64_t global_device_id,
    zuku::DeviceList mesh, std::shared_ptr<MultiMeshCompiler> compiler,
    const std::vector<ScalarArgument>& scalars,
    const std::vector<StoreHandle>& inputs,
    const std::vector<StoreHandle>& outputs,
    zuku::Future<zuku::ArrayTile>& temp, MultiMeshExecuteOptions options) {
  VLOG(3) << "Creating execute task " << compiler->Name() << " on slice "
          << mesh;

  for (const auto& input : inputs) {
    if (input.impl->offloaded) {
      throw std::runtime_error(
          absl::StrCat(input.impl->name, " still offloaded, cannot run task ",
                       compiler->Name()));
    }
    input.impl->last_op = LastOp::Execute;
    VLOG(3) << compiler->Name() << " has input " << input.unique_id << " "
            << input.impl->name << " with shape " << input.impl->shape;
    if (mesh.Contains(global_device_id) && !input.impl->has_tile) {
      throw std::runtime_error(
          absl::StrCat("input ", input.impl->name, " has no local tile"));
    }
  }

  for (const auto& output : outputs) {
    output.impl->last_op = LastOp::Execute;
    VLOG(3) << compiler->Name() << " has output " << output.unique_id << " "
            << output.impl->name << " with shape " << output.impl->shape;
    if (mesh.Contains(global_device_id) && !output.impl->has_tile) {
      throw std::runtime_error(
          absl::StrCat("input ", output.impl->name, " has no local tile"));
    }
  }

  if (compute_hashes_) {
    operation_hash_ =
        absl::HashOf(operation_hash_, kCreateExecuteTask,
                     name_to_fingerprint_[compiler->Name()],
                     compiler->LaunchSize(), mesh, inputs, outputs);
  }
}

void MockZukuExecuteContext::RunAfterAllTasks(int64_t local_device_id,
                                              std::function<void()> on_done) {}

}  // namespace xla
