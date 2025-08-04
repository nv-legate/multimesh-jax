/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_inline_explicit_tasks.h"

#include <optional>

#include "xla/pjrt/multimesh/pycallback_types.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/json_utils.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/service/call_inliner.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"

namespace xla {
namespace {

// Returns a task config for the `json` node.
// `context` gives a debug description for errors.
absl::StatusOr<multimesh::TaskOptions> GetTaskOptions(
    const Json::Value& json, const std::string& context) {
  TF_ASSIGN_OR_RETURN(auto name,
                      GetTaskValue<std::string>(json, context, "name"));
  TF_ASSIGN_OR_RETURN(auto devices, GetTaskValue<std::vector<int64_t>>(
                                        json, context, "devices"));

  multimesh::TaskOptions options{
      .name = std::move(name),
      .devices = std::move(devices),
  };

  auto autosharding_json = json.get("autosharding", Json::Value::null);
  if (!autosharding_json.isNull()) {
    TF_ASSIGN_OR_RETURN(options.dims, GetTaskValue<std::vector<int64_t>>(
                                          autosharding_json, context, "dims"));
    TF_ASSIGN_OR_RETURN(options.axes,
                        GetTaskValue<std::vector<std::string>>(
                            autosharding_json, context, "device_axes"));
    TF_ASSIGN_OR_RETURN(
        options.logical_axes,
        (GetTaskValue<std::vector<
             std::pair<std::string, std::string>>>)(autosharding_json, context,
                                                    "logical_axes"));
  }
  return options;
}

}  // namespace

absl::StatusOr<bool> MpmdInlineExplicitTasks::InlineExplicitTasks(
    HloComputation* computation) {
  bool changed = false;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->IsCustomCall("MultiMeshTask")) {
      VLOG(5) << "coloring explicit task instruction " << instruction->name();

      changed = true;

      auto json = GetJsonValue(instruction->raw_backend_config_string().data(),
                               instruction->raw_backend_config_string().size());
      if (!json.ok()) {
        return InvalidArgumentStrCat(instruction->name(),
                                     " has bad json config");
      }

      TF_ASSIGN_OR_RETURN(
          auto task_options,
          GetTaskOptions(*json, std::string(instruction->name())));

      auto dl = [&]() -> absl::StatusOr<zuku::DeviceList> {
        if (task_options.devices.empty()) {
          return partition_->Devices();
        }
        return CreateDeviceList(task_options.devices);
      }();

      if (!task_options.name.has_value()) {
        return InvalidArgumentStrCat("task on ", instruction->name(),
                                     " has no name");
      }

      std::string task_name = *task_options.name;
      TF_ASSIGN_OR_RETURN(const std::string color,
                          partition_->AllocateColor(task_name, *std::move(dl),
                                                    std::move(task_options)));

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
