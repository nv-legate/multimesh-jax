#include "legate_xla_common.h"
#include <set>

namespace legate_xla {

void CreateCompileTask(TaskArgHold<LegateCompiler> *compiler);

void CreateExecuteTask(TaskArgHold<LegateCompiler> *compiler,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       std::vector<std::function<void()>> *on_done);

StoreHandle CreateStore(const legate_xla::Shape &shape,
                        std::optional<std::string> name = std::nullopt);

void CopyDeviceToDevice(const StoreHandle &store, const void *src, size_t size,
                        size_t num_local_devices);

StoreHandle Reshard(const StoreHandle &handle,
                    const std::vector<size_t> &tile_shape);

std::set<int> GetLocalDevices(int my_node);

void SliceLocalShards(const StoreHandle &handle,
                      std::vector<void *> &local_shard,
                      const std::vector<size_t> &devices, size_t shard_id);

void CreateStoreFromHostBufferTask(const void *data, uint64_t num_bytes,
                                   StoreHandle &output,
                                   std::function<void()> on_done);

void BufferFromHostBuffer(BufferFromHostBufferAction *action,
                          StoreHandle output, int device, int num_devices,
                          bool blocking = false);

void PrintMachineConfig();

void Synchronize(const StoreHandle &store);

void Destroy(StoreHandle &store);

void StartLegate();

void StopLegate();

} // namespace legate_xla
