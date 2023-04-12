#include "xla_task.h"

namespace legate_xla {

Legion::Logger log_xla("legate.xla");

/*static*/ legate::TaskRegistrar& LegateXla::get_registrar()
{
  static legate::TaskRegistrar registrar;
  return registrar;
}

}
