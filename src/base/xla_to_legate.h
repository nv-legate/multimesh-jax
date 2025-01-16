#include "legate_xla_common.h"
#include <optional>
#include <set>
#include <string>

#include <src/zuku/shape.h>

namespace legate_xla {

struct ExecuteOptions {
  std::optional<std::string> name{std::nullopt};
  bool strict_ordering{true};
  std::optional<int> priority;
};

struct CreateStoreConfig {
  std::optional<std::string> name{std::nullopt};
  bool allocate_from_cache{false};
  std::optional<int64_t> min_cache_size{std::nullopt};
};

void CreateCompileTask(int64_t local_device_id,
                       std::shared_ptr<LegateCompiler> compiler);

void CreateExecuteTask(int64_t run_id, int64_t local_device_id,
                       int64_t global_device_id, zuku::DeviceList mesh,
                       std::shared_ptr<LegateCompiler> compiler,
                       const std::vector<ScalarArgument> &scalars,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       const BufferHandle &temp_buffer, ExecuteOptions = {});

void RunAfterAllTasks(int64_t local_device_id, std::function<void()> on_done);

void Free(int64_t local_device_id, legate_xla::StoreHandle handle,
          bool keep_in_cache);

void ClearStoreCache(int64_t local_device_id);

void OffloadDtoH(int64_t local_device_id,
                 const std::vector<StoreHandle> &to_offload,
                 const std::vector<StoreHandle> &pipelined,
                 const std::string &task_name);

void OffloadHtoD(int64_t local_device_id,
                 const std::vector<StoreHandle> &to_offload,
                 const std::string &task_name);

StoreHandle CreateStore(int64_t local_device_id, int64_t global_device_id,
                        zuku::ShardedShape shape,
                        CreateStoreConfig config = {});

BufferHandle CreateBuffer(int64_t local_device_id, int64_t global_device_id,
                          int64_t size);

void Reshard(int64_t local_device_id, int64_t global_device_id,
             const StoreHandle &src, const StoreHandle &dst);

void *SliceLocalShard(int64_t local_device_id, const StoreHandle &handle);

void StartTimer(const std::string &name);

void StopTimer(const std::string &name);

struct Shard {
  const void *data;
  int64_t local_device_id;
  std::vector<int64_t> shape_index;
  size_t size;
};

zuku::ShardedShape GetStoreShardedShape(const StoreHandle &handle);

StoreHandle
AssembleShards(int64_t local_device_id, int64_t global_device_id,
               zuku::ShardedShape shape, legate_xla::Shard shard,
               std::shared_ptr<LegateStream> stream,
               std::optional<StoreHandle> existing_store = std::nullopt);

std::set<int> GetLocalDevices();

bool IsGpu();

bool HasLocalShard(const legate_xla::StoreHandle &handle);

void StoreBufferAction(int64_t local_device_id, BufferAction *actions,
                       const StoreHandle &store, bool blocking);

void Rename(StoreHandle &handle, std::string name);

void FenceCompilation();

void FenceExecution();

void Destroy(StoreHandle &store);

void StartLegate();

void StopLegate();

} // namespace legate_xla
