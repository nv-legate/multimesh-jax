#include "legate_xla_common.h"
#include <optional>
#include <set>
#include <string>

#include <src/zuku/shape.h>

namespace legate_xla {

void CreateCompileTask(int64_t local_device_id,
                       std::shared_ptr<LegateCompiler> compiler);

void CreateExecuteTask(int64_t run_id, int64_t local_device_id,
                       zuku::DeviceList mesh,
                       std::shared_ptr<LegateCompiler> compiler,
                       const std::vector<ScalarArgument> &scalars,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       const BufferHandle &temp_buffer,
                       std::vector<std::function<void()>> *on_done);

void OffloadDtoH(const StoreHandle &src, const StoreHandle &dst);

StoreHandle CreateStore(int64_t local_device_id, int64_t global_device_id,
                        zuku::ShardedShape shape,
                        std::optional<std::string> name = std::nullopt);

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

StoreHandle
AssembleShards(int64_t local_device_id, int64_t global_device_id,
               zuku::ShardedShape shape, legate_xla::Shard shard,
               std::shared_ptr<LegateStream> stream,
               std::optional<StoreHandle> existing_store = std::nullopt);

std::set<int> GetLocalDevices();

bool IsGpu();

void StoreBufferAction(int64_t local_device_id, BufferAction *actions,
                       const StoreHandle &store, bool blocking);

void FenceCompilation();

void Destroy(StoreHandle &store);

void StartLegate();

void StopLegate();

} // namespace legate_xla
