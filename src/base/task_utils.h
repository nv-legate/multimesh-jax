#include "legate.h"

namespace legate_xla {

struct TaskConfig {
  legate::mapping::ProcessorRange device_id_range;
  int32_t task_id;
  int32_t num_tasks;
  int32_t local_proc_id;
  uint32_t min_node;
  uint32_t my_node;
};

TaskConfig get_task_config(const legate::TaskContext &context);

} // namespace legate_xla
