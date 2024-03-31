#include "task_utils.h"
#include "legate_xla_common.h"

using namespace legate;

namespace {

Legion::Logger log_xla("legate.xla_utils");

}

namespace legate_xla {

TaskConfig get_task_config(const TaskContext &context) {
  auto device_id_range = context.machine().processor_range();
  auto task_id = static_cast<int32_t>(context.get_task_index()[0]);
  return {
      .device_id_range = device_id_range,
      .task_id = task_id,
      .num_tasks = (int32_t)device_id_range.count(),
      .local_device_id = static_cast<int32_t>((device_id_range.low + task_id) %
                                              device_id_range.per_node_count),
      .my_device_id = static_cast<int32_t>(device_id_range.low + task_id),
      .min_node = device_id_range.low / device_id_range.per_node_count,
      .my_node =
          (device_id_range.low + task_id) / device_id_range.per_node_count,
  };
}

bool BlockingExecution() {
  if (const char *blocking = getenv("LEGATE_XLA_BLOCKING")) {
    return std::atoi(blocking);
  }
  return false;
}

} // namespace legate_xla
