#include "legate_xla_common.h"

namespace legate_xla {

void CreateCompileTask(LegateCompiler *compiler);

void CreateExecuteTask(LegateExecutable *executable,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       std::vector<std::function<void()>> *on_done);

StoreHandle CreateStore(const legate_xla::Shape &shape);

void CreateStoreFromHostBufferTask(const void *data, uint64_t num_bytes,
                                   StoreHandle output,
                                   std::function<void()> on_done);

void Synchronize(StoreHandle store);

void CopyStoreToHostSync(StoreHandle input,
                         std::function<void(const void *)> copy_func);

void Destroy(StoreHandle store);

void InitLegate();

} // namespace legate_xla
