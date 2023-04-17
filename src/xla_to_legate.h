#include "legate_xla_common.h"

namespace legate_xla {

void CreateCompileTask(LegateCompiler* compiler);

void CreateExecuteTask(LegateExecutable* executable,
  const std::vector<StoreHandle>& inputs,
  const std::vector<StoreHandle>& outputs);

StoreHandle CreateStore(const legate_xla::Shape& shape);


StoreHandle CreateStoreFromHostBuffer(
  const legate_xla::Shape& shape, const void* data, std::function<void()> on_done);

void InitLegate();

}