#include "xla/pjrt/legate/loop_scheduler.h"

#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/util.h"

namespace xla {

#if 0
absl::Status ScheduleGpipe(const SchedulingUnit& unit,
                           std::vector<ScheduledTask>& tasks) {
  int64_t num_iterations = unit.config().num_iterations;
  int64_t unrolling = num_iterations;
  if (unit.config().unrolling.has_value()) {
    unrolling = *unit.config().unrolling;
  }

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "Scheduling loop unit with " << num_iterations << " iterations, "
            << unit.tasks().size() << " tasks";
    for (const auto* task : unit.tasks()) {
      VLOG(5) << " -> " << task->Compiler()->Name();
    }
  }

  for (int64_t iter_offset = 0; iter_offset < num_iterations;
       iter_offset += unrolling) {
    int64_t end = std::min(iter_offset + unrolling, num_iterations);
    for (const auto& task : unit.tasks()) {
      for (int32_t iter = iter_offset; iter < end; ++iter) {
        bool last = iter == (num_iterations - 1);
        tasks.push_back(
            ScheduledTask{.task = task,
                          .iter = iter,
                          .last = last,
                          .devices = task->Compiler()->IterationSlice(iter)});
      }
    }
  }

  return absl::OkStatus();
}

absl::Status ScheduleWavefront(int64_t total_num_devices,
                               absl::Span<const ExecutableTask* const> tasks,
                               const LoopConfig& loop_config,
                               std::vector<ScheduledTask>& schedule) {
  if (tasks.empty()) {
    return;
  }

  const int last_iteration = loop_config.num_iterations - 1;

  VLOG(5) << "Have loop config num_stages="
          << loop_config.num_stages.value_or(-1)
          << ", interleave=" << loop_config.interleave.value_or(-1);

  const auto [num_groups, num_pipeline_tasks] = [&] {
    absl::flat_hash_map<zuku::DeviceList, int64_t> unique_groups;
    int num_pipeline_tasks = 0;
    for (auto&& task : tasks) {
      auto devices = task->Compiler()->IterationSlice(0);
      if (devices.size() != total_num_devices) {
        // don't count tasks over the global set of devices
        unique_groups[devices] += 1;
        ++num_pipeline_tasks;
      }
    }
    return std::make_pair(std::max<int>(1, unique_groups.size()),
                          num_pipeline_tasks);
  }();

  const int next_minibatch_delay = [&,
                                    num_pipeline_tasks = num_pipeline_tasks] {
    // if explicitly specified, create the schedule based on a
    // fixed number of stages
    if (loop_config.num_stages.has_value()) {
      return *loop_config.num_stages;
    }
    // otherwise guess at the number of stages
    return std::max<int>(1, (num_pipeline_tasks + 1) / 2);
  }();

  const int num_stages = tasks.size();
  const int minibatch_size = num_groups;
  const int num_microbatches = loop_config.num_iterations;
  const int num_minibatches =
      (loop_config.num_iterations + num_groups - 1) / num_groups;
  const int num_wavefronts =
      num_stages + num_minibatches * next_minibatch_delay;

  VLOG(3) << "Have num_groups=" << num_groups
          << ", minibatch_size=" << minibatch_size
          << ", num_minibatches=" << num_minibatches
          << ", num_wavefronts=" << num_wavefronts
          << ", minibatch_delay=" << next_minibatch_delay;

  for (int wf = 0; wf < num_wavefronts; ++wf) {
    for (int stage = num_stages - 1; stage >= 0; --stage) {
      const int tick = wf - stage;
      const int minibatch = tick / next_minibatch_delay;
      const int minibatch_offset = tick % next_minibatch_delay;
      if (tick >= 0 && minibatch_offset < minibatch_size) {
        const int microbatch = minibatch * minibatch_size + minibatch_offset;
        if (microbatch < num_microbatches) {
          const ExecutableTask* task = tasks[stage];
          auto machine_slice = task->Compiler()->IterationSlice(microbatch);
          VLOG(3) << "Adding wf=" << wf << " microbatch=" << microbatch
                  << " to stage=" << stage << ", slice=["
                  << machine_slice.start() << "..." << machine_slice.stop()
                  << ")"
                  << " " << task->Compiler()->Name();

          schedule.push_back(
              ScheduledTask{.task = task,
                            .iter = microbatch,
                            .last = (microbatch == last_iteration),
                            .devices = std::move(machine_slice)});
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status SchedulePrefetchWavefront(int64_t total_num_devices,
                                       const SchedulingUnit& unit,
                                       int num_to_unroll,
                                       std::vector<ScheduledTask>& schedule) {
  // gpipe schedule the first tasks to get everyone started and to avoid
  // delays later in the pipeline
  const int last_iteration = unit.config().num_iterations - 1;
  for (int task = 0; task < num_to_unroll; ++task) {
    auto* first_task = unit.tasks()[task];
    VLOG(3) << "fully unrolling first task " << first_task->Compiler()->Name()
            << " to prefetch initial activations for all iterations";
    for (int iter = 0; iter < unit.config().num_iterations; ++iter) {
      schedule.push_back(ScheduledTask{
          .task = first_task,
          .iter = iter,
          .last = (iter == last_iteration),
          .devices = first_task->Compiler()->IterationSlice(iter)});
    }
  }
  return ScheduleWavefront(
      total_num_devices,
      {&unit.tasks()[num_to_unroll], unit.tasks().size() - num_to_unroll},
      unit.config(), schedule);
}

absl::Status SchedulePrefetchWavefront(int64_t total_num_devices,
                                       const SchedulingUnit& unit,
                                       std::vector<ScheduledTask>& schedule) {
  if (unit.tasks().empty()) {
    return absl::OkStatus();
  }

  if (unit.tasks().size() == 1) {
    // not interesting unless at least two
    return ScheduleGpipe(unit, schedule);
  }

  auto unroll_task = [](const ExecutableTask& task) {
    return task.LoopIncrement();
  };

  int num_to_unroll = 0;
  while (num_to_unroll < unit.tasks().size() &&
         unroll_task(*unit.tasks()[num_to_unroll])) {
    ++num_to_unroll;
  }

  if (num_to_unroll > 0) {
    return SchedulePrefetchWavefront(total_num_devices, unit, num_to_unroll,
                                     schedule);
  }
  return ScheduleWavefront(total_num_devices, unit.tasks(), unit.config(),
                           schedule);
}

absl::Status ScheduleWavefront(int64_t total_num_devices,
                               const SchedulingUnit& unit,
                               std::vector<ScheduledTask>& schedule) {
  if (unit.tasks().empty()) {
    return absl::OkStatus();
  }

  if (unit.tasks().size() == 1) {
    // not interesting unless at least two
    return ScheduleGpipe(unit, schedule);
  }

  if (unit.tasks()[0]->LoopIncrement()) {
    return SchedulePrefetchWavefront(total_num_devices, unit,
                                     /*num_to_unroll=*/1, schedule);
  }

  return ScheduleWavefront(total_num_devices, unit.tasks(), unit.config(),
                           schedule);
}

absl::StatusOr<std::vector<ScheduledTask>> ScheduleLoops(
    int64_t num_global_devices, const std::vector<SchedulingUnit>& units) {
  std::vector<ScheduledTask> tasks;
  size_t num_tasks = 0;
  for (const auto& unit : units) {
    if (unit.has_loop_config()) {
      // the +1 is because there is a increment task
      // at the beginning of the loop
      num_tasks += unit.config().num_iterations * (unit.tasks().size() + 1);
    } else {
      num_tasks += unit.tasks().size();
    }
  }
  tasks.reserve(num_tasks);

  for (const auto& unit : units) {
    if (unit.has_loop_config()) {
      switch (unit.config().schedule) {
        case LoopConfig::Schedule::kFillDrain:
          TF_RETURN_IF_ERROR(ScheduleGpipe(unit, tasks));
          break;
        case LoopConfig::Schedule::kPrefetchWavefront:
          TF_RETURN_IF_ERROR(
              SchedulePrefetchWavefront(num_global_devices, unit, tasks));
          break;
        case LoopConfig::Schedule::k1F1B:
        case LoopConfig::Schedule::kWavefront:
          TF_RETURN_IF_ERROR(
              ScheduleWavefront(num_global_devices, unit, tasks));
          break;
      }
    } else {
      for (const auto* task : unit.tasks()) {
        VLOG(5) << "Scheduling single task " << task->Compiler()->Name();
        tasks.push_back(ScheduledTask{
            .task = task, .devices = task->Compiler()->IterationSlice(0)});
      }
    }
  }

  return std::move(tasks);
}

#endif

absl::Status ScheduleWavefront(const HloPartition& partition,
                               const std::vector<std::vector<HloInstruction*>>& tasks,
                               const LoopConfig& loop_config,
                               std::vector<HloInstruction*>& schedule) {
  if (tasks.empty()) {
    return;
  }

  const int last_iteration = loop_config.num_iterations - 1;

  VLOG(5) << "Have loop config num_stages="
          << loop_config.num_stages.value_or(-1)
          << ", interleave=" << loop_config.interleave.value_or(-1);

  const int64_t total_num_devices = partition.TotalDevices();
  const auto [num_groups, num_pipeline_tasks] = [&] {
    absl::flat_hash_map<zuku::DeviceList, int64_t> unique_groups;
    int num_pipeline_tasks = 0;
    for (auto&& task : tasks[0]){
      auto color = Color(task);
      auto devices = partition.DevicesForColor(*color);
      if (devices.size() != total_num_devices) {
        // don't count tasks over the global set of devices
        unique_groups[devices] += 1;
        ++num_pipeline_tasks;
      }
    }
    return std::make_pair(std::max<int>(1, unique_groups.size()),
                          num_pipeline_tasks);
  }();

  const int next_minibatch_delay = [&,
                                    num_pipeline_tasks = num_pipeline_tasks] {
    // if explicitly specified, create the schedule based on a
    // fixed number of stages
    if (loop_config.num_stages.has_value()) {
      return *loop_config.num_stages;
    }
    // otherwise guess at the number of stages
    return std::max<int>(1, (num_pipeline_tasks + 1) / 2);
  }();

  const int num_stages = tasks[0].size();
  const int minibatch_size = num_groups;
  const int num_microbatches = loop_config.num_iterations;
  const int num_minibatches =
      (loop_config.num_iterations + num_groups - 1) / num_groups;
  const int num_wavefronts =
      num_stages + num_minibatches * next_minibatch_delay;

  VLOG(3) << "Have num_groups=" << num_groups
          << ", minibatch_size=" << minibatch_size
          << ", num_minibatches=" << num_minibatches
          << ", num_wavefronts=" << num_wavefronts
          << ", minibatch_delay=" << next_minibatch_delay;

  for (int wf = 0; wf < num_wavefronts; ++wf) {
    for (int stage = num_stages - 1; stage >= 0; --stage) {
      const int tick = wf - stage;
      const int minibatch = tick / next_minibatch_delay;
      const int minibatch_offset = tick % next_minibatch_delay;
      if (tick >= 0 && minibatch_offset < minibatch_size) {
        const int microbatch = minibatch * minibatch_size + minibatch_offset;
        if (microbatch < num_microbatches) {
          schedule.push_back(tasks[microbatch][stage]);
          if (VLOG_IS_ON(3)){
            auto color = Color(schedule.back());
            auto devices = partition.DevicesForColor(*color);
            VLOG(3) << "Adding wf=" << wf << " microbatch=" << microbatch
                    << " to stage=" << stage << ", " << schedule.back()->called_computations()[0]->name()
                    << " across " << devices;
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<HloInstruction*>> ScheduleGpipe(const LoopConfig& config, const std::vector<std::vector<HloInstruction*>>& tasks)
{
  std::vector<HloInstruction*> schedule;
  schedule.reserve(config.num_iterations * tasks[0].size());
  const int64_t tasks_per_iter = tasks[0].size();
  for (int64_t task_index = 0; task_index < tasks_per_iter; ++task_index) {
    for (int iter = 0; iter < config.num_iterations; ++iter) {
      schedule.push_back(tasks[iter][task_index]);
    }
  }
  return schedule;
}

absl::StatusOr<std::vector<HloInstruction*>> SchedulePrefetchWavefront(const HloPartition& partition,
                                       const LoopConfig& config,
                                       int num_to_unroll,
                                       const std::vector<std::vector<HloInstruction*>>& tasks) {
  // gpipe schedule the first tasks to get everyone started and to avoid
  // delays later in the pipeline
  std::vector<HloInstruction*> schedule;
  for (int task = 0; task < num_to_unroll; ++task) {
    VLOG(3) << "fully unrolling first task " << tasks[0][task]->called_computations()[0]->name()
            << " to prefetch initial activations for all iterations";
    for (int iter = 0; iter < config.num_iterations; ++iter) {
      schedule.push_back(tasks[iter][task]);
    }
  }

  std::vector<std::vector<HloInstruction*>> remaining_tasks(tasks.size());
  for (int64_t iter=0; iter < config.num_iterations; ++iter){
    remaining_tasks[iter].reserve(tasks[iter].size() - num_to_unroll);
    for (int64_t task=num_to_unroll; task < tasks[iter].size(); ++task){
      remaining_tasks[iter].push_back(tasks[iter][task]);
    }
  }

  TF_RETURN_IF_ERROR(ScheduleWavefront(
    partition,
    remaining_tasks, config, schedule));

  return schedule;
}

absl::StatusOr<std::vector<HloInstruction*>> ScheduleWavefront(
  const HloPartition& partition, const LoopConfig& config, const std::vector<std::vector<HloInstruction*>>& tasks)
{
  if (tasks.empty() || tasks[0].empty()){
    return std::vector<HloInstruction*>{};
  }

  if (tasks[0].size() == 1){
    return ScheduleGpipe(config, tasks);
  }


  auto unroll_task = [&](HloInstruction* call) {
    auto color = Color(call);
    return partition.IsLoopIncrementColor(*color);
  };

  int64_t num_to_unroll = 0;
  if (unroll_task(tasks[0][0])) {
    ++num_to_unroll;
  }

  return SchedulePrefetchWavefront(partition, config,
                                    num_to_unroll, tasks);
}

absl::StatusOr<std::vector<HloInstruction*>> SchedulePrefetchWavefront(const HloPartition& partition, const LoopConfig& config, const std::vector<std::vector<HloInstruction*>>& tasks){

  if (tasks.empty() || tasks[0].empty()) {
    return std::vector<HloInstruction*>{};
  }

  if (tasks[0].size() == 1) {
    // not interesting unless at least two
    return ScheduleGpipe(config, tasks);
  }

  auto unroll_task = [&](HloInstruction* call) {
    auto color = Color(call);
    return partition.IsLoopIncrementColor(*color);
  };

  int num_to_unroll = 0;
  while (num_to_unroll < tasks[0].size() &&
         unroll_task(tasks[0][num_to_unroll])) {
    ++num_to_unroll;
  }

  if (num_to_unroll > 0) {
    return SchedulePrefetchWavefront(partition, config, num_to_unroll,
                                     tasks);
  }
  return ScheduleWavefront(partition, config, tasks);
}

absl::StatusOr<std::vector<HloInstruction*>> ScheduleLoops(
  const HloPartition& partition, const LoopConfig& config, const std::vector<std::vector<HloInstruction*>>& tasks)
{
  switch (config.schedule) {
    case LoopConfig::Schedule::kFillDrain:
      return ScheduleGpipe(config, tasks);
    case LoopConfig::Schedule::kPrefetchWavefront:
      return SchedulePrefetchWavefront(partition, config, tasks);
    case LoopConfig::Schedule::k1F1B:
    case LoopConfig::Schedule::kWavefront:
      return ScheduleWavefront(partition, config, tasks);
  }
}



}  // namespace xla