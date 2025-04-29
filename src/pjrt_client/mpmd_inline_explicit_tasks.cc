/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_inline_explicit_tasks.h"

#include <optional>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/json_utils.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"
#include "xla/service/call_inliner.h"
#include "xla/service/tuple_simplifier.h"

namespace xla {
namespace {

struct TaskConfig {
  std::string name;
  std::vector<int64_t> devices;
  std::optional<LogicalShardingContext> autosharding;
};

// Returns a task config for the `json` node.
// `context` gives a debug description for errors.
absl::StatusOr<TaskConfig> GetTaskConfig(const Json::Value& json,
                                         const std::string& context) {
  TF_ASSIGN_OR_RETURN(auto name,
                      GetTaskValue<std::string>(json, context, "name"));
  TF_ASSIGN_OR_RETURN(auto devices, GetTaskValue<std::vector<int64_t>>(
                                        json, context, "devices"));
  TF_ASSIGN_OR_RETURN(
      std::optional<int64_t> loop_submesh_size,
      GetOptionalTaskValue<int64_t>(json, context, "loop_submesh_size"));
  TF_ASSIGN_OR_RETURN(
      bool loop_submesh_reverse,
      GetOptionalTaskValue(json, context, "loop_submesh_reverse", false));

  auto autosharding_json = json.get("autosharding", Json::Value::null);
  std::optional<LogicalShardingContext> autosharding;
  if (!autosharding_json.isNull()) {
    TF_ASSIGN_OR_RETURN(autosharding, GetLogicalShardingContext(
                                          autosharding_json, context, devices));
  }

  if (loop_submesh_size.has_value()) {
    autosharding->loop_submesh =
        LoopDependentSubmesh({.task_mesh_size = *loop_submesh_size,
                              .global_mesh_start = devices.front(),
                              .global_mesh_stop = devices.back() + 1,
                              .reverse = loop_submesh_reverse});
  }

  return TaskConfig{
      .name = std::move(name),
      .devices = std::move(devices),
      .autosharding = std::move(autosharding),
  };
}
}  // namespace

absl::StatusOr<bool> MpmdInlineExplicitTasks::InlineExplicitTasks(
    HloComputation* computation) {
  bool changed = false;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->IsCustomCall("LegateTask")) {
      VLOG(5) << "coloring explicit task instruction " << instruction->name();

      changed = true;

      auto json = GetJsonValue(instruction->raw_backend_config_string().data(),
                               instruction->raw_backend_config_string().size());
      if (!json.ok()) {
        return InvalidArgumentStrCat(instruction->name(),
                                     " has bad json config");
      }

      TF_ASSIGN_OR_RETURN(
          TaskConfig config,
          GetTaskConfig(*json, std::string(instruction->name())));

      auto dl = [&]() -> absl::StatusOr<zuku::DeviceList> {
        if (config.devices.empty()) {
          return partition_->Devices();
        }
        return CreateDeviceList(config.devices);
      }();

      std::shared_ptr<LogicalShardingContext> context;
      if (config.autosharding.has_value()) {
        context = std::make_shared<LogicalShardingContext>(
            *std::move(config.autosharding));
      }

      TF_ASSIGN_OR_RETURN(const std::string color,
                          partition_->AllocateColor(config.name, *std::move(dl),
                                                    std::move(context)));

      for (auto* sub : instruction->called_computations()[0]->instructions()) {
        VLOG(5) << sub->name() << " assigned color=" << color
                << " from task custom-call";
        AssignColor(sub, color);
      }
      // turn the task into a regular call and inline
      auto* call_to_inline =
          computation->AddInstruction(HloInstruction::CreateCall(
              instruction->shape(), instruction->operands(),
              instruction->called_computations()[0]));

      TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(call_to_inline));

      AssignColor(instruction, color);
    } else {
      for (auto* comp : instruction->called_computations()) {
        TF_ASSIGN_OR_RETURN(bool comp_changed, InlineExplicitTasks(comp));
        changed |= comp_changed;
      }
    }
  }

  return changed;
}

absl::StatusOr<bool> MpmdInlineExplicitTasks::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  TF_ASSIGN_OR_RETURN(bool changed,
                      InlineExplicitTasks(module->entry_computation()));
  if (changed) {
    CallInliner inliner;
    TF_ASSIGN_OR_RETURN(bool _, inliner.Run(module));

    TupleSimplifier simplifier;
    TF_ASSIGN_OR_RETURN(_, simplifier.Run(module));
  }

  return changed;
}

}  // namespace xla
