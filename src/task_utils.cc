#include "task_utils.h"

using namespace legate;

namespace legate_xla {

TaskConfig get_task_config(TaskContext& context){
#ifdef LEGATE_XLA_PYTHON_PROTOTYPE
  auto device_id_range = context.machine_desc().processor_range();
  auto task_id = static_cast<int32_t>(context.get_task_index()[0]);
  return {
    .device_id_range = device_id_range,
    .task_id         = task_id,
    .num_tasks       = (int32_t)device_id_range.count(),
    .local_proc_id =
      static_cast<int32_t>((device_id_range.lo + task_id) % device_id_range.per_node_count),
    .min_node = device_id_range.lo / device_id_range.per_node_count,
    .my_node  = (device_id_range.lo + task_id) / device_id_range.per_node_count,
  };
#else
  return {
    .task_id = 0,
    .num_tasks = 1,
    .local_proc_id = 0,
    .min_node = 0,
    .my_node = 0,
  };
#endif
}

}
