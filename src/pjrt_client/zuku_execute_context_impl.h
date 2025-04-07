/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_ZUKU_EXECUTE_CONTEXT_IMPL_H_
#define XLA_PJRT_LEGATE_ZUKU_EXECUTE_CONTEXT_IMPL_H_

#include "xla/pjrt/legate/zuku_execute_context.h"

namespace xla {

class ZukuExecuteContextImpl final : public ZukuExecuteContext {
 public:
  static std::shared_ptr<ZukuExecuteContext> Create(zuku::RealmConfig config);

  void CreateExecuteTask(int64_t run_id, int64_t local_device_id,
                         int64_t global_device_id, zuku::DeviceList devices,
                         std::shared_ptr<LegateCompiler> compiler,
                         const std::vector<ScalarArgument> &scalars,
                         const std::vector<StoreHandle> &inputs,
                         const std::vector<StoreHandle> &outputs,
                         zuku::Future<zuku::ArrayTile> &temp_buffer,
                         LegateExecuteOptions options) override;

  void CreateCompileTask(int64_t local_device_id,
                         std::shared_ptr<LegateCompiler> compiler) override;

  void RunAfterAllTasks(int64_t local_device_id,
                        std::function<void()> on_done) override;

  void Free(int64_t local_device_id, StoreHandle handle,
            bool keep_in_cache) override;

  void Clear();

  void ClearStoreCache(int64_t local_device_id) override;

  void OffloadDtoH(int64_t local_device_id,
                   const std::vector<StoreHandle> &to_offload,
                   const std::vector<StoreHandle> &pipelined,
                   const std::string &task_name) override;

  void OffloadHtoD(int64_t local_device_id,
                   const std::vector<StoreHandle> &to_offload,
                   const std::string &task_name) override;

  zuku::Future<zuku::ArrayTile> CreateBuffer(int64_t local_device_id,
                                             int64_t global_device_id,
                                             int64_t size) override;

  void Reshard(int64_t local_device_id, int64_t global_device_id,
               const StoreHandle &src, const StoreHandle &dst) override;

  void *SliceLocalShard(int64_t local_device_id,
                        const StoreHandle &handle) override;

  void StartTimer(const std::string &name) override;

  void StopTimer(const std::string &name) override;

  zuku::ShardedShape GetStoreShardedShape(const StoreHandle &handle) override;

  StoreHandle AssembleShards(
      int64_t local_device_id, int64_t global_device_id,
      zuku::ShardedShape shape, Shard shard,
      std::shared_ptr<LegateStream> stream,
      std::optional<StoreHandle> existing_store = std::nullopt) {
    return AssembleShardsImpl(local_device_id, global_device_id,
                              std::move(shape), std::move(shard),
                              std::move(stream), std::move(existing_store));
  }

  std::set<int> GetLocalDevices() override;

  bool IsGpu() override;

  bool HasLocalShard(const StoreHandle &handle) override;

  void StoreBufferAction(int64_t local_device_id, BufferAction *actions,
                         const StoreHandle &store, bool blocking) override;

  void Rename(StoreHandle &handle, std::string name) override;

  void FenceCompilation() override;

  void Destroy(StoreHandle &store) override;

  void OpenWindow() override;

  void CloseWindow() override;

  static void ClearAllContexts();

  ~ZukuExecuteContextImpl();

 private:
  explicit ZukuExecuteContextImpl(Realm::Runtime rt);

  StoreHandle AssembleShardsImpl(
      int64_t local_device_id, int64_t global_device_id,
      zuku::ShardedShape shape, Shard shard,
      std::shared_ptr<LegateStream> stream,
      std::optional<StoreHandle> existing_store) override;

  StoreHandle CreateStoreImpl(int64_t local_device_id, int64_t global_device_id,
                              zuku::ShardedShape shape,
                              CreateStoreConfig config) override;

  zuku::Processor LocalProcessor(int64_t local_index);

  void SetLastExecuteEvent(const zuku::Processor &p, Realm::Event ev);

  void SetLastControlEvent(const zuku::Processor &p, Realm::UserEvent ev);

  zuku::store_vector<zuku::ShardedArray> GetStores(
      const std::vector<StoreHandle> &handles);

  zuku::store_variant_vector<zuku::ShardedArray> GetStores(
      const std::vector<StoreHandle> &handles,
      const std::set<int64_t> &output_ids);

  using std_timer = decltype(std::chrono::steady_clock::now());
  std::unordered_map<std::string, zuku::Future<std_timer>> pending_timers_;

  std::deque<Realm::Event> window_markers_;

  std::vector<Realm::Event> last_execute_events_;
  std::vector<Realm::UserEvent> last_control_events_;
  std::vector<zuku::ArrayCache> host_caches_;
  std::vector<zuku::ArrayCache> device_caches_;
  std::set<Realm::Event> pending_compilation_events_;
  static std::set<ZukuExecuteContextImpl *> all_contexts_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_ZUKU_EXECUTE_CONTEXT_IMPL_H_
