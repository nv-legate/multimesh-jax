#include "legate_xla_common.h"
#include <optional>
#include <set>
#include <string>

namespace legate_xla {

void CreateCompileTask(TaskArgHold<LegateCompiler> *compiler);

void CreateExecuteTask(TaskArgHold<LegateCompiler> *compiler,
                       const std::vector<ScalarArgument> &scalar,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       std::vector<std::function<void()>> *on_done);

void CopyDeviceToDevice(const StoreHandle &store, const void *src, size_t size,
                        size_t num_local_devices);

StoreHandle CreateStore(const legate_xla::Shape &shape,
                        std::optional<std::string> name = std::nullopt);

StoreHandle Reshard(const StoreHandle &handle,
                    const std::vector<int64_t> &tile_shape);

void SliceLocalShards(const StoreHandle &handle,
                      std::vector<void *> &local_shards,
                      std::pair<int64_t, int64_t> slice);

void StartTimer(const std::string &name);

void StopTimer(const std::string &name);

struct Shard {
  const void *data;
  int64_t local_device_id;
  std::vector<int64_t> shape_index;
  size_t size;
};

StoreFuture
AssembleShards(const legate_xla::Shape &logical_shape,
               const std::vector<legate_xla::Shard> &local_shards,
               std::pair<int64_t, int64_t> slice,
               TaskArgHold<LegateStream> *stream_hold,
               std::optional<StoreHandle> existing_store = std::nullopt);

std::set<int> GetLocalDevices(int my_node);

bool IsGpu();

struct BufferActionConfig {
  bool blocking{false};
  std::pair<int, int> machine_slice;
};

void StoreBufferAction(const std::vector<BufferAction *> &actions,
                       const StoreHandle &store,
                       BufferActionConfig config = {});

void BeginTrace(uint32_t id);

void EndTrace(uint32_t id);

void SetScalar(legate_xla::StoreHandle store, size_t launch_size,
               int32_t scalar);

void Synchronize(const StoreHandle &store);

void Destroy(StoreHandle &store);

void StartLegate();

void StopLegate();

} // namespace legate_xla
