#include "task_utils.h"

using namespace legate;

namespace legate_xla {

TaskConfig get_task_config(TaskContext &context) {
  auto device_id_range = context.machine_desc().processor_range();
  auto task_id = static_cast<int32_t>(context.get_task_index()[0]);
  return {
      .device_id_range = device_id_range,
      .task_id = task_id,
      .num_tasks = (int32_t)device_id_range.count(),
      .local_proc_id = static_cast<int32_t>((device_id_range.low + task_id) %
                                            device_id_range.per_node_count),
      .min_node = device_id_range.low / device_id_range.per_node_count,
      .my_node =
          (device_id_range.low + task_id) / device_id_range.per_node_count,
  };
}

} // namespace legate_xla
