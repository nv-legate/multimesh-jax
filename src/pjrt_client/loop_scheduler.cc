/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/loop_scheduler.h"

#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_loop.h"
#include "xla/util.h"

namespace xla {

static inline std::vector<int64_t> GetIndicesToUnroll(
    const HloPartition& partition,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  std::vector<int64_t> indices_to_unroll;

  auto unroll_task = [&](HloInstruction* call) {
    auto color = Color(call);
    return partition.IsLoopIncrementColor(*color);
  };

  for (int64_t task_index = 0; task_index < tasks[0].size(); ++task_index) {
    if (unroll_task(tasks[0][task_index])) {
      indices_to_unroll.push_back(task_index);
    }
  }

  return indices_to_unroll;
}

static inline std::vector<std::vector<HloInstruction*>>
UnrollLoopIncrementTasks(const std::vector<std::vector<HloInstruction*>>& tasks,
                         const std::vector<int64_t>& indices_to_unroll,
                         std::vector<std::vector<HloInstruction*>>& schedule) {
  const int64_t num_to_unroll = indices_to_unroll.size();
  const int64_t num_iterations = tasks.size();

  if (num_to_unroll == 0) return tasks;

  // This function unrolls the loop increment tasks for the unrollable tasks
  // (side effect) tasks and returns the remaining tasks (return value)
  for (int64_t task_index : indices_to_unroll) {
    VLOG(3) << "fully unrolling task "
            << tasks[0][task_index]->called_computations()[0]->name();
    for (int iter = 0; iter < num_iterations; ++iter) {
      schedule[0].push_back(tasks[iter][task_index]);
    }
  }

  std::vector<std::vector<HloInstruction*>> remaining_tasks(tasks.size());
  for (int64_t iter = 0; iter < num_iterations; ++iter) {
    remaining_tasks[iter].reserve(tasks[iter].size() - num_to_unroll);
    int64_t task_index = 0;
    int64_t loop_task_index = 0;
    while (task_index < tasks[iter].size()) {
      if (task_index == indices_to_unroll[loop_task_index]) {
        ++loop_task_index;
        ++task_index;
        continue;
      }
      remaining_tasks[iter].push_back(tasks[iter][task_index]);
      ++task_index;
    }
  }

  return remaining_tasks;
}

absl::Status ScheduleWavefront(
    const HloPartition& partition,
    const std::vector<std::vector<HloInstruction*>>& tasks,
    const LoopConfig& loop_config,
    std::vector<std::vector<HloInstruction*>>& schedule) {
  if (tasks.empty()) {
    return;
  }

  if (schedule.size() != 1) {
    return InvalidArgumentStrCat("Schedule must be a single vector");
  }

  const int last_iteration = loop_config.num_iterations - 1;

  VLOG(5) << "Have loop config num_stages="
          << loop_config.num_stages.value_or(-1)
          << ", interleave=" << loop_config.interleave.value_or(-1);

  const int64_t total_num_devices = partition.TotalDevices();
  const auto [num_groups, num_pipeline_tasks] = [&] {
    absl::flat_hash_map<zuku::DeviceList, int64_t> unique_groups;
    int num_pipeline_tasks = 0;
    for (auto&& task : tasks[0]) {
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
          schedule[0].push_back(tasks[microbatch][stage]);
          if (VLOG_IS_ON(3)) {
            auto color = Color(schedule[0].back());
            auto devices = partition.DevicesForColor(*color);
            VLOG(3) << "Adding wf=" << wf << " microbatch=" << microbatch
                    << " to stage=" << stage << ", "
                    << schedule[0].back()->called_computations()[0]->name()
                    << " across " << devices;
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<HloInstruction*>>> ScheduleGpipe(
    const LoopConfig& config,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  std::vector<std::vector<HloInstruction*>> schedule(1);
  schedule[0].reserve(config.num_iterations * tasks[0].size());
  const int64_t tasks_per_iter = tasks[0].size();
  for (int64_t task_index = 0; task_index < tasks_per_iter; ++task_index) {
    for (int iter = 0; iter < config.num_iterations; ++iter) {
      schedule[0].push_back(tasks[iter][task_index]);
    }
  }
  return schedule;
}

absl::StatusOr<std::vector<std::vector<HloInstruction*>>>
ScheduleCustomSchedule(
    const std::vector<std::vector<std::pair<int64_t, std::string>>>&
        custom_schedule,
    const std::vector<std::vector<HloInstruction*>>& tasks,
    std::vector<std::vector<HloInstruction*>>& schedule) {
  const int64_t num_microbatches = tasks.size();
  const int64_t num_custom_schedule_rows = custom_schedule.size();
  const int64_t tasks_per_iter = tasks.front().size();

  absl::flat_hash_map<std::string, int64_t> custom_schedule_task_counts;
  for (int64_t mesh_id = 0; mesh_id < num_custom_schedule_rows; ++mesh_id) {
    for (const auto& [iter, task_name] : custom_schedule[mesh_id]) {
      custom_schedule_task_counts[task_name] += 1;
    }
  }

  // Check 1: Check that each task appears in each custom schedule device row
  // exactly microbatch times
  for (const auto& [task_name, count] : custom_schedule_task_counts) {
    if (count != num_microbatches) {
      return InvalidArgumentStrCat("Task ", task_name, " appears ", count,
                                   " times in custom schedule, expected ",
                                   num_microbatches);
    }
  }

  // AND construct a map from (microbatch iter, task_name) to the task
  // HloInstruction* in the schedule
  absl::flat_hash_map<std::pair<int64_t, std::string>, HloInstruction*>
      task_map;
  for (int64_t iter = 0; iter < num_microbatches; ++iter) {
    // O(n2). Could be more efficient if we know the names match exactly.
    for (const auto& [task_name, count] : custom_schedule_task_counts) {
      for (int64_t mb_task_index = 0; mb_task_index < tasks_per_iter;
           ++mb_task_index) {
        if (absl::StrContains(ColorOrDefault(tasks[iter][mb_task_index]),
                              task_name)) {
          if (task_map.contains({iter, task_name})) {
            // Check 2: Check that each color we expect appears AT MOST ONCE in
            // the task row for each microbatch. This is the first half of
            // checking if coloring + grouping + fusion was done correctly.
            // Checking each row is fairly redundant, but we do it anyway.
            return InvalidArgumentStrCat(
                "Task ", task_name,
                " appears multiple times in microbatch row ", iter);
          }
          VLOG(5)
              << "Adding task "
              << tasks[iter][mb_task_index]->called_computations()[0]->name()
              << " with color " << ColorOrDefault(tasks[iter][mb_task_index])
              << " at iter " << iter << " and task_index " << mb_task_index;
          task_map[{iter, task_name}] = tasks[iter][mb_task_index];
        }
      }
    }
  }

  // Populate the schedule
  for (int64_t mesh_id = 0; mesh_id < num_custom_schedule_rows; ++mesh_id) {
    const std::vector<std::pair<int64_t, std::string>>& schedule_for_mesh =
        custom_schedule[mesh_id];
    for (const auto& [iter, task_name] : schedule_for_mesh) {
      HloInstruction* task_inst = task_map[{iter, task_name}];
      // Check 3: Check that each task appears in the schedule AT LEAST ONCE.
      // This is the other half of checking if coloring + grouping + fusion was
      // done correctly.
      if (task_inst == nullptr)
        return InvalidArgumentStrCat("Task ", task_name,
                                     " not found in task map!");
      VLOG(5) << "Scheduling task "
              << task_inst->called_computations()[0]->name() << " at mesh_id "
              << mesh_id << " and iter " << iter;
      schedule[mesh_id].push_back(task_inst);
    }
  }

  return schedule;
}

absl::StatusOr<std::vector<std::vector<HloInstruction*>>> ScheduleCustom(
    const HloPartition& partition, const LoopConfig& config,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  if (!config.custom_schedule.has_value()) {
    throw std::invalid_argument("Needs custom schedule or callback");
  }

  const std::vector<std::vector<std::pair<int64_t, std::string>>>
      custom_schedule = *config.custom_schedule;
  std::vector<std::vector<HloInstruction*>> schedule(custom_schedule.size());

  const std::vector<int64_t> indices_to_unroll =
      GetIndicesToUnroll(partition, tasks);
  const std::vector<std::vector<HloInstruction*>> remaining_tasks =
      UnrollLoopIncrementTasks(tasks, indices_to_unroll, schedule);

  for (int64_t stage = 0; stage < custom_schedule.size(); ++stage) {
    schedule[stage].reserve(custom_schedule[stage].size() +
                            schedule[stage].size());
  }

  return ScheduleCustomSchedule(custom_schedule, remaining_tasks, schedule);
}

absl::StatusOr<std::vector<std::vector<HloInstruction*>>>
SchedulePrefetchWavefront(
    const HloPartition& partition, const LoopConfig& config,
    const std::vector<int64_t>& indices_to_unroll,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  // gpipe schedule the first tasks to get everyone started and to avoid
  // delays later in the pipeline
  std::vector<std::vector<HloInstruction*>> schedule(1);

  const std::vector<std::vector<HloInstruction*>> remaining_tasks =
      UnrollLoopIncrementTasks(tasks, indices_to_unroll, schedule);

  TF_RETURN_IF_ERROR(
      ScheduleWavefront(partition, remaining_tasks, config, schedule));

  return schedule;
}

absl::StatusOr<std::vector<std::vector<HloInstruction*>>> ScheduleWavefront(
    const HloPartition& partition, const LoopConfig& config,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  if (tasks.empty() || tasks[0].empty()) {
    return std::vector<std::vector<HloInstruction*>>(1);
  }

  if (tasks[0].size() == 1) {
    return ScheduleGpipe(config, tasks);
  }

  const std::vector<int64_t> indices_to_unroll =
      GetIndicesToUnroll(partition, tasks);

  return SchedulePrefetchWavefront(partition, config, indices_to_unroll, tasks);
}

absl::StatusOr<std::vector<std::vector<HloInstruction*>>>
SchedulePrefetchWavefront(
    const HloPartition& partition, const LoopConfig& config,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  if (tasks.empty() || tasks[0].empty()) {
    return std::vector<std::vector<HloInstruction*>>{
        std::vector<HloInstruction*>{}};
  }

  if (tasks[0].size() == 1) {
    // not interesting unless at least two
    return ScheduleGpipe(config, tasks);
  }

  const std::vector<int64_t> indices_to_unroll =
      GetIndicesToUnroll(partition, tasks);

  if (indices_to_unroll.size() > 0) {
    return SchedulePrefetchWavefront(partition, config, indices_to_unroll,
                                     tasks);
  }
  return ScheduleWavefront(partition, config, tasks);
}

absl::StatusOr<std::vector<std::vector<HloInstruction*>>> ScheduleLoops(
    const HloPartition& partition, const LoopConfig& config,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  switch (config.schedule) {
    case LoopConfig::Schedule::kCustom:
      return ScheduleCustom(partition, config, tasks);
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
