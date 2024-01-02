#include "legate_xla_common.h"
#include <optional>
#include <set>
#include <string>

namespace legate_xla {

void CreateCompileTask(TaskArgHold<LegateCompiler> *compiler);

void CreateExecuteTask(TaskArgHold<LegateCompiler> *compiler,
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
                      const std::vector<int64_t> &devices);

struct Shard {
  const void *data;
  int64_t local_device_id;
  std::vector<int64_t> shape_index;
  size_t size;
};

StoreHandle AssembleShards(const legate_xla::Shape &shape,
                           const std::vector<Shard> &shards);

std::set<int> GetLocalDevices(int my_node);

bool IsGpu();

struct BufferActionConfig {
  bool blocking{false};
  std::optional<std::pair<int, int>> machine_slice{std::nullopt};
};

void StoreBufferAction(const std::vector<BufferAction *> &actions,
                       const StoreHandle &store,
                       BufferActionConfig config = {});

void SetScalar(legate_xla::StoreHandle handle, size_t launch_size,
               int32_t scalar);

void Synchronize(const StoreHandle &store);

void Destroy(StoreHandle &store);

void StartLegate();

void StopLegate();

} // namespace legate_xla
