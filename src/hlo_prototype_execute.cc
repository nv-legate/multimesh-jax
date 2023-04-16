#include "hlo_prototype_execute.h"
#include "executable_cache.h"
#include "hlo_executor.h"

namespace legate_xla {

/*static*/ void HLOPrototypeExecuteTask::run_executable(legate::TaskContext& context, const std::string& platform)
{
  uint64_t run_id = context.scalars()[0].value<uint64_t>();
  uint64_t hlo_id = context.scalars()[1].value<uint64_t>();
  std::string hlo_name = context.scalars()[2].value<std::string>();

  auto* executable = find_executable(hlo_id);
  HLOExecutorTask::run_executable(context, executable, run_id, /*scalar_offset=*/3);
}

/*static*/ void HLOPrototypeExecuteTask::cpu_variant(legate::TaskContext& context)
{
  run_executable(context, "cpu");
}

}