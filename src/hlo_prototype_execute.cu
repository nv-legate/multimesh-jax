#include "hlo_prototype_execute.h"
#include "executable_cache.h"
#include "hlo_executor.h"

namespace legate_xla {

/*static*/ void HLOPrototypeExecuteTask::gpu_variant(legate::TaskContext& context)
{
  run_executable(context, "gpu");
}

}