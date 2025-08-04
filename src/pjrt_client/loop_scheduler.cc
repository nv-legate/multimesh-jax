/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/loop_scheduler.h"
#include <functional>
#include <queue>

#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_loop.h"
#include "xla/util.h"

namespace xla {
namespace {

absl::flat_hash_set<int64_t> GetIndicesToUnroll(
    const HloPartition& partition,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  absl::flat_hash_set<int64_t> indices_to_unroll;

  auto unroll_task = [&](HloInstruction* call) {
    auto color = Color(call);
    return partition.IsLoopIncrementColor(*color);
  };

  for (int64_t task_index = 0; task_index < tasks[0].size(); ++task_index) {
    if (unroll_task(tasks[0][task_index])) {
      indices_to_unroll.insert(task_index);
    }
  }

  return indices_to_unroll;
}

std::vector<std::vector<HloInstruction*>> UnrollLoopIncrementTasks(
    const std::vector<std::vector<HloInstruction*>>& tasks,
    const absl::flat_hash_set<int64_t>& indices_to_unroll,
    std::vector<std::vector<HloInstruction*>>& schedule) {
  const int64_t num_to_unroll = indices_to_unroll.size();
  const int64_t num_iterations = tasks.size();

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
    for (int64_t task_index = 0; task_index < tasks[iter].size();
         ++task_index) {
      if (!indices_to_unroll.contains(task_index)) {
        remaining_tasks[iter].push_back(tasks[iter][task_index]);
      }
    }
  }

  return remaining_tasks;
}

}  // namespace

absl::StatusOr<std::vector<std::vector<HloInstruction*>>> ScheduleZeroBubble(
    const HloPartition& partition, const LoopConfig& loop_config,
    const std::vector<std::vector<HloInstruction*>>& tasks) {
  if (tasks.empty()) {
    return std::vector<std::vector<HloInstruction*>>{};
  }

  std::vector<std::vector<HloInstruction*>> reordered_tasks(tasks.size());
  for (int iter = 0; iter < tasks.size(); ++iter) {
    reordered_tasks[iter].reserve(tasks.size());
  }

  absl::flat_hash_map<HloInstruction*, int64_t> critical_stage_numbers;
  int64_t num_pipeline_tasks = 0;
  for (int64_t stage = 0; stage < tasks[0].size(); ++stage) {
    if (partition.IsCritical(tasks[0][stage])) {
      for (int iter = 0; iter < tasks.size(); ++iter) {
        HloInstruction* task = tasks[iter][stage];
        critical_stage_numbers[task] = stage;
        reordered_tasks[iter].push_back(task);
        VLOG(5) << "critical stage " << task->to_apply()->name()
                << " pushed back in reordering for iteration " << iter;
      }
      ++num_pipeline_tasks;
    }
  }

  struct NonCriticalStage {
    int64_t partner_stage_number;
    int64_t stage;
  };
  struct CompareNonCriticalStage {
    bool operator()(const NonCriticalStage& lhs,
                    const NonCriticalStage& rhs) const {
      return lhs.partner_stage_number < rhs.partner_stage_number;
    }
  };

  // we want to reverse order of the non-critical stages
  // BWD 3, BWD 2, BWD 1, BWD 0, GRAD 0, GRAD 1, GRAD 2, GRAD 3
  std::priority_queue<NonCriticalStage, std::vector<NonCriticalStage>,
                      CompareNonCriticalStage>
      non_critical_stages;
  // determine criticality based on the LAST microbatch
  for (int64_t stage = 0; stage < tasks.back().size(); ++stage) {
    auto* instruction = tasks.back()[stage];
    if (partition.IsNonCritical(instruction)) {
      int64_t critical_partner_stage = 0;
      for (auto* gte_operand : instruction->operands()) {
        for (auto* call_operand : gte_operand->operands()) {
          if (critical_stage_numbers.contains(call_operand)) {
            critical_partner_stage = std::max(
                critical_partner_stage, critical_stage_numbers[call_operand]);
          }
        }
      }
      VLOG(5) << "non-critical stage " << stage << " consumes critical stage "
              << critical_partner_stage;
      non_critical_stages.emplace(
          NonCriticalStage{critical_partner_stage, stage});
    }
  }

  while (!non_critical_stages.empty()) {
    NonCriticalStage nc_stage = non_critical_stages.top();
    non_critical_stages.pop();
    VLOG(5) << "non-critical stage "
            << tasks[0][nc_stage.stage]->to_apply()->name()
            << " pushed back in reordering";
    for (int64_t iter = 0; iter < tasks.size(); ++iter) {
      reordered_tasks[iter].push_back(tasks[iter][nc_stage.stage]);
    }
  }

  VLOG(5) << "Have loop config num_stages="
          << loop_config.num_stages.value_or(-1)
          << ", interleave=" << loop_config.interleave.value_or(-1);

  std::vector<HloInstruction*> order;
  const int64_t total_num_devices = partition.TotalDevices();
  const auto num_groups = [&] {
    absl::flat_hash_map<zuku::DeviceList, int64_t> unique_groups;
    for (auto&& task : tasks[0]) {
      auto color = Color(task);
      auto devices = partition.DevicesForColor(*color);
      if (devices.size() != total_num_devices) {
        // don't count tasks over the global set of devices
        unique_groups[devices] += 1;
      }
    }
    return std::max<int>(1, unique_groups.size());
  }();

  const int next_minibatch_delay = (num_pipeline_tasks + 1) / 2;
  const int num_stages = tasks[0].size();
  const int minibatch_size = num_groups;
  const int num_minibatches =
      (loop_config.num_iterations + num_groups - 1) / num_groups;
  const int num_wavefronts =
      num_stages + num_minibatches * next_minibatch_delay;

  struct StageGroup {
    int64_t wf;
    int64_t microbatch;
    int64_t stage;
  };

  // use > to prioritize earlier stages
  struct CompareStageGroup {
    bool operator()(const StageGroup& lhs, const StageGroup& rhs) const {
      if (lhs.wf != rhs.wf) {
        return lhs.wf > rhs.wf;
      }
      if (lhs.microbatch != rhs.microbatch) {
        return lhs.microbatch > rhs.microbatch;
      }
      return lhs.stage > rhs.stage;
    }
  };

  std::priority_queue<StageGroup, std::vector<StageGroup>, CompareStageGroup>
      queue;
  for (int minibatch = 0; minibatch < num_minibatches; ++minibatch) {
    const int64_t time_offset = minibatch * next_minibatch_delay;
    for (int64_t mb = 0; mb < minibatch_size; ++mb) {
      const int64_t microbatch = minibatch * minibatch_size + mb;
      for (int64_t stage = 0; stage < tasks[0].size(); ++stage) {
        const int64_t wf = time_offset + mb + stage;
        VLOG(5) << "emplacing wf=" << wf << " microbatch=" << microbatch
                << " stage=" << stage;
        queue.emplace(StageGroup{wf, microbatch, stage});
      }
    }
  }

  VLOG(3) << "Have num_groups=" << num_groups
          << ", minibatch_size=" << minibatch_size
          << ", num_minibatches=" << num_minibatches
          << ", num_wavefronts=" << num_wavefronts
          << ", minibatch_delay=" << next_minibatch_delay;

  absl::flat_hash_map<zuku::DeviceList, double> device_timer;
  absl::flat_hash_map<int64_t, double> microbatch_timer;
  absl::flat_hash_map<int64_t, zuku::DeviceList> last_devices_for_microbatch;
  while (!queue.empty()) {
    StageGroup group = std::move(queue.top());
    queue.pop();
    order.push_back(reordered_tasks[group.microbatch][group.stage]);
    auto devices = partition.DevicesForInstruction(order.back());
    const auto [time, buffer] = [&] {
      auto& devices_time = device_timer[devices];
      if (partition.IsCritical(tasks[0][group.stage])) {
        auto& microbatch_time = microbatch_timer[group.microbatch];
        double comm_time = [&] {
          if (last_devices_for_microbatch.contains(group.microbatch) &&
              devices != last_devices_for_microbatch[group.microbatch]) {
            return 0.1;
          }
          return 0.0;
        }();
        last_devices_for_microbatch[group.microbatch] = devices;
        auto time = std::max(devices_time, microbatch_time + comm_time);
        auto buffer = time - microbatch_time;
        devices_time = time + 1;
        microbatch_time = time + 1;
        return std::make_pair(time, buffer);
      }
      return std::make_pair(devices_time++, -1.0);
    }();

    if (VLOG_IS_ON(5)) {
      std::cerr << "popped wf=" << group.wf
                << " microbatch=" << group.microbatch
                << " stage=" << group.stage << " " << devices
                << " at t=" << time << ", buffer=" << buffer
                << ", call=" << order.back()->name() << std::endl;
    }
  }
  return std::vector<std::vector<HloInstruction*>>{std::move(order)};
}

absl::Status ScheduleWavefront(
    const HloPartition& partition,
    const std::vector<std::vector<HloInstruction*>>& tasks,
    const LoopConfig& loop_config,
    std::vector<std::vector<HloInstruction*>>& schedule) {
  if (tasks.empty()) {
    return absl::OkStatus();
  }

  if (schedule.size() != 1) {
    return InvalidArgumentStrCat("Schedule must be a single vector");
  }

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

  for (int64_t task_index = 0; task_index < tasks_per_iter; ++task_index) {
    VLOG(5) << "task " << tasks[0][task_index]->name() << " "
            << tasks[0][task_index]->called_computations()[0]->name() << " "
            << ColorOrDefault(tasks[0][task_index])
            << " to be custom scheduled";
  }

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

  const absl::flat_hash_set<int64_t> indices_to_unroll =
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
    const absl::flat_hash_set<int64_t>& indices_to_unroll,
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

  const absl::flat_hash_set<int64_t> indices_to_unroll =
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

  const absl::flat_hash_set<int64_t> indices_to_unroll =
      GetIndicesToUnroll(partition, tasks);

  if (!indices_to_unroll.empty()) {
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
    case LoopConfig::Schedule::kZeroBubbleH2:
      return ScheduleZeroBubble(partition, config, tasks);
    case LoopConfig::Schedule::k1F1B:
    case LoopConfig::Schedule::kWavefront:
      return ScheduleWavefront(partition, config, tasks);
  }
}

}  // namespace xla
