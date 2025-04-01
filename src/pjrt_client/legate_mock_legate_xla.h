#ifndef XLA_PJRT_LEGATE_LEGATE_MOCK_LEGATE_XLA_H_
#define XLA_PJRT_LEGATE_LEGATE_MOCK_LEGATE_XLA_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/legate/zuku_execute_context.h"

namespace xla {

enum Operation {
  kCopyDeviceToDevice,
  kCreateStore,
  kCreateBuffer,
  kReshard,
  kStoreBufferAction,
  kDestroy,
  kCreateExecuteTask,
  kCreateCompileTask,
  kAssembleShards,
  kOffloadDtoH,
  kOffloadHtoD,
};

enum MemoryKind { kHOST, kDEVICE };

enum class LastOp { Execute, Reshard, OffloadHtoD, OffloadDtoH, Created };

struct MockZukuExecuteContext;

class MockArrayCache {
 public:
  void Allocate(int64_t local_device_id, const zuku::ShardedShape &shape,
                MemoryKind kind, MockZukuExecuteContext &context);

  void Free(const zuku::ShardedShape &shape, MemoryKind kind,
            MockZukuExecuteContext &context);

  int64_t NumAllocations() const { return num_allocated_; }

  void Clear(int64_t local_device_id, MemoryKind kind,
             const zuku::ShardedShape &shape, MockZukuExecuteContext &context);

  bool Full() const { return num_allocated_ == num_available_; }

  bool Empty() const { return num_available_ == 0; }

 private:
  MemoryKind kind_;
  int64_t device_;
  int64_t chunk_size_;
  int64_t num_allocated_;
  int64_t num_available_;
};

struct MemoryPool {
  int64_t allocated{0};
  int64_t high_watermark{0};
};

class MockZukuExecuteContext final : public ZukuExecuteContext {
 public:
  void Reset();

  void SetComputeHashes(bool flag) { compute_hashes_ = flag; }

  int64_t OperationHash() const { return operation_hash_; }

  int64_t HostBytesHighWatermark(int64_t local_device_id);

  int64_t DeviceBytesHighWatermark(int64_t local_device_id);

  void SetCompileModules(bool flag) { compile_modules_ = flag; }

  void AllocateFromDevice(MemoryKind kind, int64_t device, int64_t bytes);

  void DeallocateFromDevice(MemoryKind kind, int64_t device, int64_t bytes);

  void OpenWindow() override;

  void CloseWindow() override;

  void CreateExecuteTask(int64_t run_id, int64_t local_device_id,
                         int64_t global_device_id, zuku::DeviceList mesh,
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

 private:
  StoreHandle CreateStoreImpl(int64_t local_device_id, int64_t global_device_id,
                              zuku::ShardedShape shape,
                              CreateStoreConfig config) override;

  StoreHandle AssembleShardsImpl(
      int64_t local_device_id, int64_t global_device_id,
      zuku::ShardedShape shape, Shard shard,
      std::shared_ptr<LegateStream> stream,
      std::optional<StoreHandle> existing_store) override;

  int64_t store_id_counter_{0};

  int64_t operation_hash_{0};

  bool compute_hashes_{false};

  bool compile_modules_{false};

  absl::flat_hash_map<int64_t, absl::flat_hash_map<MemoryKind, MemoryPool>>
      device_memories_;

  absl::flat_hash_map<std::string, std::string> name_to_fingerprint_;
  absl::flat_hash_map<std::string, int> created_store_counts_;
  absl::flat_hash_map<int64_t,
                      absl::flat_hash_map<zuku::ShardedShape, MockArrayCache>>
      device_caches_;

  absl::flat_hash_map<int64_t,
                      absl::flat_hash_map<zuku::ShardedShape, MockArrayCache>>
      host_caches_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_LEGATE_MOCK_LEGATE_XLA_H_