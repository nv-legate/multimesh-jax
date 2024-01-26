#include "task_utils.h"

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

void TaskWaiter::Wait() {
  log_xla.debug() << "TaskWaiter::Signal: waiting on " << this;
  std::unique_lock lk(m_);
  cv_.wait(lk, [&] { return ready_; });
}

int64_t TaskWaiter::Signal() {
  int64_t remainining = num_pending_.fetch_add(int64_t(-1));
  log_xla.debug() << "TaskWaiter::Signal: signaling " << this << " with "
                  << remainining << " pending";
  if (remainining == 1) {
    // off by one, 1 means this was the last one to run
    {
      std::lock_guard lk(m_);
      ready_ = true;
    }
    cv_.notify_one();
  }
  return remainining - 1;
}

} // namespace legate_xla
