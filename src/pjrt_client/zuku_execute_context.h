/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_ZUKU_EXECUTE_CONTEXT_H_
#define XLA_PJRT_MULTIMESH_ZUKU_EXECUTE_CONTEXT_H_

#include <optional>
#include <set>
#include <string>

#include "src/zuku/tiled_array.h"
#include "xla/pjrt/multimesh/mm_computation.h"
#include "xla/pjrt/multimesh/scalar_argument.h"
#include "xla/pjrt/multimesh/store_handle_fwd.h"

namespace xla {

struct Shard {
  const void* data;
  int64_t local_device_id;
  size_t size;
};

struct CreateStoreConfig {
  std::optional<std::string> name{std::nullopt};
  bool allocate_from_cache{false};
  std::optional<int64_t> min_cache_size{std::nullopt};
};

struct MultiMeshExecuteOptions {
  std::optional<std::string> name{std::nullopt};
  bool strict_ordering{true};
  std::optional<int> priority;
};

class ZukuExecuteContext {
 public:
  virtual void CreateExecuteTask(int64_t run_id, int64_t local_device_id,
                                 int64_t global_device_id,
                                 zuku::DeviceList mesh,
                                 std::shared_ptr<MultiMeshCompiler> compiler,
                                 const std::vector<ScalarArgument>& scalars,
                                 const std::vector<StoreHandle>& inputs,
                                 const std::vector<StoreHandle>& outputs,
                                 zuku::Future<zuku::ArrayTile>& temp_buffer,
                                 MultiMeshExecuteOptions options) = 0;

  virtual void CreateCompileTask(
      int64_t local_device_id, std::shared_ptr<MultiMeshCompiler> compiler) = 0;

  virtual void RunAfterAllTasks(int64_t local_device_id,
                                std::function<void()> on_done) = 0;

  virtual void Free(int64_t local_device_id, StoreHandle handle,
                    bool keep_in_cache) = 0;

  virtual void ClearStoreCache(int64_t local_device_id) = 0;

  virtual void OffloadDtoH(int64_t local_device_id,
                           const std::vector<StoreHandle>& to_offload,
                           const std::vector<StoreHandle>& pipelined,
                           const std::string& task_name) = 0;

  virtual void OffloadHtoD(int64_t local_device_id,
                           const std::vector<StoreHandle>& to_offload,
                           const std::string& task_name) = 0;

  virtual zuku::Future<zuku::ArrayTile> CreateBuffer(int64_t local_device_id,
                                                     int64_t global_device_id,
                                                     int64_t size) = 0;

  virtual void Reshard(int64_t local_device_id, int64_t global_device_id,
                       const StoreHandle& src, const StoreHandle& dst) = 0;

  virtual void* SliceLocalShard(int64_t local_device_id,
                                const StoreHandle& handle) = 0;

  virtual void MarkProfile(const std::string& name) = 0;

  virtual void StartTimer(const std::string& name) = 0;

  virtual void StopTimer(const std::string& name) = 0;

  virtual zuku::ShardedShape GetStoreShardedShape(
      const StoreHandle& handle) = 0;

  StoreHandle AssembleShards(
      int64_t local_device_id, int64_t global_device_id,
      zuku::ShardedShape shape, Shard shard,
      std::shared_ptr<MultiMeshStream> stream,
      std::optional<StoreHandle> existing_store = std::nullopt) {
    return AssembleShardsImpl(local_device_id, global_device_id,
                              std::move(shape), std::move(shard),
                              std::move(stream), std::move(existing_store));
  }

  StoreHandle CreateStore(int64_t local_device_id, int64_t global_device_id,
                          zuku::ShardedShape shape,
                          CreateStoreConfig config = {}) {
    return CreateStoreImpl(local_device_id, global_device_id, std::move(shape),
                           std::move(config));
  }

  virtual void OpenWindow() = 0;

  virtual void CloseWindow() = 0;

  virtual std::set<int> GetLocalDevices() = 0;

  virtual bool IsGpu() = 0;

  virtual bool HasLocalShard(const StoreHandle& handle) = 0;

  virtual void Rename(StoreHandle& handle, std::string name) = 0;

  virtual void FenceCompilation() = 0;

  virtual void Destroy(StoreHandle& store) = 0;

 private:
  virtual StoreHandle CreateStoreImpl(int64_t local_device_id,
                                      int64_t global_device_id,
                                      zuku::ShardedShape shape,
                                      CreateStoreConfig config) = 0;

  virtual StoreHandle AssembleShardsImpl(
      int64_t local_device_id, int64_t global_device_id,
      zuku::ShardedShape shape, Shard shard,
      std::shared_ptr<MultiMeshStream> stream,
      std::optional<StoreHandle> existing_store) = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_ZUKU_EXECUTE_CONTEXT_H_
