/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_buffer_scheduling_name.h"

#include <algorithm>
#include <queue>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/legate_sharding.h"
#include "xla/pjrt/legate/mpmd_instruction.h"

namespace xla {

struct BufferUse {
  std::string name;
  int64_t available;
};

struct CompareTimes {
  bool operator()(const BufferUse& lhs, const BufferUse& rhs) const {
    return lhs.available > rhs.available;
  }
};

class BufferPool {
 public:
  std::optional<std::string> AllocateAtTime(int64_t time,
                                            const zuku::ShardedShape& shape) {
    if (uses.empty()) {
      return std::nullopt;
    }

    auto& next = uses.top();
    VLOG(5) << "trying to allocate from buffer pool at t=" << time
            << ", top=" << uses.top().available << " for shape " << shape;
    if (uses.top().available <= time) {
      std::string buffer{std::move(uses.top().name)};
      uses.pop();
      return buffer;
    }
    return std::nullopt;
  }

  void FreeAtTime(int64_t time, std::string name) {
    uses.emplace(BufferUse{std::move(name), time});
  }

  void ClearUpToTime(int64_t time) {
    while (!uses.empty() && uses.top().available <= time) {
      uses.pop();
    }
  }

 private:
  zuku::ShardedShape shape;
  std::priority_queue<BufferUse, std::vector<BufferUse>, CompareTimes> uses;
};

class BufferAllocator {
 public:
  void FreeAtTime(int64_t time, std::string name) {
    const auto& shape = shapes_[name];
    VLOG(5) << "free " << name << " at t=" << time << ", shape=" << shape;
    pools_[shape].FreeAtTime(time, std::move(name));
  }

  std::string AllocateAtTime(const zuku::ShardedShape& shape, int64_t time,
                             HloInstruction* instruction) {
    VLOG(5) << "allocate " << instruction->name() << " at t=" << time
            << ", shape=" << shape;
    auto& pool = pools_[shape];
    auto buffer = pool.AllocateAtTime(time, shape);
    std::string name = [&] {
      if (buffer.has_value()) {
        return *std::move(buffer);
      }
      return std::string(instruction->name());
    }();
    shapes_[name] = shape;
    return name;
  }

 private:
  absl::flat_hash_map<zuku::ShardedShape, BufferPool> pools_;
  absl::flat_hash_map<std::string, zuku::ShardedShape> shapes_;
};

absl::StatusOr<bool> MpmdAssignBufferSchedulingName::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  absl::flat_hash_map<zuku::DeviceList, int64_t> time_on_mesh;

  // a counter for tracking how many times an instruction has been an operand
  // to identify when we are visiting the last use
  absl::flat_hash_map<const HloInstruction*, int64_t> times_visited;

  absl::flat_hash_map<const HloInstruction*, int64_t> times_available;

  BufferAllocator allocator;

  HloSharding replicated = HloSharding::Replicate();

  auto free_operand = [&](int64_t time_done,
                          HloInstruction* operand) -> absl::Status {
    if (operand->IsCustomCall("SliceOffset")) {
      return absl::OkStatus();
    }
    if (operand->opcode() != HloOpcode::kParameter) {
      if (operand->metadata().scheduling_name().empty()) {
        return InvalidArgumentStrCat(operand->name(),
                                     " was not assigned a scheduling name in "
                                     "MpmdAssignBufferSchedulingName");
      }
      const int64_t num_visits = ++times_visited[operand];
      if (num_visits == operand->user_count()) {
        allocator.FreeAtTime(time_done, operand->metadata().scheduling_name());
      }
    }
    return absl::OkStatus();
  };

  auto visit_output = [&](int64_t time_started, HloInstruction* output,
                          const zuku::DeviceList& devices) -> absl::Status {
    VLOG(5) << "visiting output " << output->name();
    if (!output->metadata().scheduling_name().empty()) {
      VLOG(5) << output->name() << " already assigned to "
              << output->metadata().scheduling_name();
      // already assigned
      return absl::OkStatus();
    }
    // make sure this is not a root instruction and needs a temp buffer
    // assignment
    if (absl::c_any_of(output->users(), [&](HloInstruction* i) {
          return i == module->entry_computation()->root_instruction();
        })) {
      output->set_metadata_scheduling_name(output->name());
    } else {
      TF_ASSIGN_OR_RETURN(
          zuku::ShardedShape user_shape,
          XlaShapeToLegateShape(output->shape(), devices,
                                output->sharding_or_default(replicated)));
      std::string name =
          allocator.AllocateAtTime(user_shape, time_started, output);
      VLOG(5) << "assigned scheduling name " << name << " to temp "
              << output->name() << " at t=" << time_started;
      output->set_metadata_scheduling_name(std::move(name));
    }
    return absl::OkStatus();
  };

  for (auto* instruction :
       module->entry_computation()->parameter_instructions()) {
    instruction->set_metadata_scheduling_name(instruction->name());
  }

  // at this point, the module should have been flattened into the entry
  // computation and all the loops should have been unrolled
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      auto color = Color(instruction);
      if (!color.has_value()) {
        return InvalidArgumentStrCat(
            "instruction ", instruction->name(),
            " was not assigned a color prior to buffer assignment");
      }
      auto devices = partition_->DevicesForColor(*color);

      int64_t max_operand_time = time_on_mesh[devices];
      for (auto* operand : instruction->operands()) {
        max_operand_time = std::max(times_available[operand], max_operand_time);
      }
      // for now we use a stupid model that increments the time by 1 for every
      // operation
      const int64_t time_done = max_operand_time + 1;

      absl::flat_hash_set<std::string> output_names;
      for (auto* user : instruction->users()) {
        times_available[user] = time_done;
        TF_RETURN_IF_ERROR(visit_output(max_operand_time, user, devices));
        output_names.insert(user->metadata().scheduling_name());
      }
      for (auto* operand : instruction->operands()) {
        // only free the operand if it is not an output of the call
        if (!output_names.contains(operand->metadata().scheduling_name())) {
          TF_RETURN_IF_ERROR(free_operand(time_done, operand));
        }
      }

      VLOG(5) << instruction->name() << " started at " << max_operand_time
              << " and advanced time to t=" << time_done << " on " << devices;
      time_on_mesh[devices] = time_done;
    } else if (instruction->IsCustomCall("Reshard")) {
      auto color = Color(instruction);
      if (!color.has_value()) {
        return InvalidArgumentStrCat(
            "instruction ", instruction->name(),
            " was not assigned a color prior to buffer assignment");
      }
      auto devices = partition_->DevicesForColor(*color);
      HloInstruction* operand = instruction->mutable_operand(0);
      const int64_t time_started = times_available[operand];
      // we reserve 2 time units for communication to ensure that we never block
      // on anti-dependencies where an output buffer is assigned to a task, but
      // can't be used because it is waiting for a send to finish
      const int64_t time_done = time_started + 2;
      TF_RETURN_IF_ERROR(visit_output(time_started, instruction, devices));
      if (operand->opcode() != HloOpcode::kParameter) {
        auto operand_color = Color(operand);
        if (!operand_color.has_value()) {
          return InvalidArgumentStrCat(
              "instruction ", operand->name(),
              " was not assigned a color prior to buffer assignment");
        }
        TF_RETURN_IF_ERROR(free_operand(time_done, operand));
      }
      times_available[instruction] = time_done;
    }
  }

  // buffers always get assigned
  return true;
}

}  // namespace xla
