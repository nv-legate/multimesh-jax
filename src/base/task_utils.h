#include "legate.h"
#include <condition_variable>

namespace legate_xla {

struct TaskConfig {
  legate::mapping::ProcessorRange device_id_range;
  int32_t task_id;
  int32_t num_tasks;
  int32_t local_device_id;
  int32_t my_device_id;
  uint32_t min_node;
  uint32_t my_node;
};

TaskConfig get_task_config(const legate::TaskContext &context);

class TaskWaiter {
public:
  TaskWaiter(int64_t num_tasks) : ready_(false), num_pending_{num_tasks} {}

  void Wait();

  void Signal();

private:
  bool ready_;
  std::atomic<int64_t> num_pending_;
  std::condition_variable cv_;
  std::mutex m_;
};

} // namespace legate_xla
