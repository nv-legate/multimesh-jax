#include "legate_xla_common.h"

namespace legate_xla {

void CreateCompileTask(LegateCompiler *compiler);

void CreateExecuteTask(LegateExecutable *executable,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       std::vector<std::function<void()>> *on_done);

StoreHandle CreateStore(const legate_xla::Shape &shape);

StoreHandle Reshard(const StoreHandle &handle,
                    const std::vector<size_t> &tile_shape);

void SliceLocalShards(const StoreHandle &handle,
                      std::vector<void *> &local_shards);

void CreateStoreFromHostBufferTask(const void *data, uint64_t num_bytes,
                                   StoreHandle &output,
                                   std::function<void()> on_done);

void Synchronize(const StoreHandle &store);

void Destroy(StoreHandle &store);

void StartLegate();

void StopLegate();

} // namespace legate_xla
