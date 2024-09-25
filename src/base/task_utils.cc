#include "task_utils.h"
#include "legate_xla_common.h"

namespace legate_xla {



bool BlockingExecution() {
  if (const char *blocking = getenv("LEGATE_XLA_BLOCKING")) {
    return std::atoi(blocking);
  }
  return false;
}

} // namespace legate_xla
