#include "hlo_prototype_distributed_shutdown.h"

namespace legate_xla {

/*static*/ void
HloPrototypeDistributedShutdownTask::gpu_variant(legate::TaskContext &context) {
  shutdown_distributed(context);
}

} // namespace legate_xla