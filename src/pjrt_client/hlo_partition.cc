/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/hlo_partition.h"

#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "src/zuku/mesh.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/regexp.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/pjrt/legate/json_utils.h"
#include "xla/pjrt/legate/legate_sharding.h"
#include "xla/pjrt/legate/logical_sharding_context.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"
#include "xla/util.h"

namespace xla {

namespace {

bool enable_metadata_name_tasks = true;
bool enable_recomputation = false;

constexpr absl::string_view kLoopIncrementColor = "loop_increment";
constexpr absl::string_view kJvpTransposeMetadata = "transpose(jvp";
constexpr char kLegateTaskCustomCallTarget[] = "LegateTask";
constexpr char kMicrobatchCustomCallTarget[] = "Microbatch";
constexpr char kMicrobatchInitCustomCallTarget[] = "MicrobatchInit";
constexpr char kMicrobatchSliceCustomCallTarget[] = "MicrobatchSlice";
constexpr char kAutoShardingCustomCallTarget[] = "AutoSharding";

const absl::flat_hash_set<std::string> kLegateCustomCallTargets = {
    kLegateTaskCustomCallTarget, kMicrobatchCustomCallTarget,
    kMicrobatchInitCustomCallTarget, kMicrobatchSliceCustomCallTarget,
    kAutoShardingCustomCallTarget};

// Struct containing metadata about an explicit task
// defined at the Python level.
// `color` is a unique partition label assigned
// in the application code for distinguishing tasks.
struct TaskConfig {
  std::string name;
  std::vector<int64_t> devices;
  std::optional<LogicalShardingContext> autosharding;
};

}  // namespace

absl::StatusOr<Json::Value> GetJson(HloInstruction* instruction) {
  auto json = GetJsonValue(instruction->raw_backend_config_string().data(),
                           instruction->raw_backend_config_string().size());
  if (!json.ok()) {
    return InvalidArgumentStrCat("could not parse backend config for: ",
                                 json.status().message());
  }
  return std::move(json);
}

absl::StatusOr<Json::Value> GetJson(const HloInstructionProto& instr) {
  auto json = GetJsonValue(instr.backend_config().data(),
                           instr.backend_config().size());
  if (!json.ok()) {
    return InvalidArgumentStrCat("could not parse backend config for ",
                                 instr.name(), ": ", json.status().message());
  }
  return std::move(json);
}

// Struct holding all the metadata for matching implicit tasks
// and mapping matched names to physical submeshes
struct NamedTaskContext {
  std::unique_ptr<re2::RE2> matcher;
  HloPartition::name_and_devices_fxn callback;
  std::vector<int64_t> dims;
  std::vector<std::string> device_axes;
  std::vector<std::pair</*logical=*/std::string, /*device=*/std::string>>
      logical_axes;
  bool backprop{false};
};

// Static helper for returning implicit task map mapping
// matched names to their task metadata
std::vector<NamedTaskContext>& NamedTaskContexts() {
  static auto* named_task_contexts = new std::vector<NamedTaskContext>;
  return *named_task_contexts;
}

absl::StatusOr<std::shared_ptr<LogicalShardingContext>>
MakeLogicalShardingContext(zuku::DeviceList devices,
                           const NamedTaskContext& implicit_task_config) {
  IndexVector<int64_t> dims{implicit_task_config.dims.begin(),
                            implicit_task_config.dims.end()};
  auto context =
      std::make_shared<LogicalShardingContext>(LogicalShardingContext{
          .devices = std::move(devices),
          .dims = implicit_task_config.dims,
          .device_axes = implicit_task_config.device_axes,
          .logical_axes = implicit_task_config.logical_axes,
      });
  return std::move(context);
}

absl::Status HloPartition::Recolor(
    const absl::flat_hash_map<std::string, std::string>& color_map) {
  absl::flat_hash_map<std::string, ColorConfig> new_configs;
  for (auto& [new_color, old_color] : color_map) {
    VLOG(5) << this << " recoloring " << old_color << " to " << new_color;
    auto iter = colors_.find(old_color);
    if (iter == colors_.end()) {
      return InvalidArgumentStrCat("cannot remap color ", old_color,
                                   ", has not been created");
    }
    new_configs[new_color] = iter->second;
  }
  colors_ = std::move(new_configs);
  return absl::OkStatus();
}

std::shared_ptr<LogicalShardingContext> HloPartition::GetLogicalShardingContext(
    const std::string& color) const {
  return ConfigForColor(color).logical_sharding_context;
}

zuku::DeviceList HloPartition::DefaultPerTaskMesh(
    const std::string& color) const {
  const auto& config = ConfigForColor(color);
  if (config.logical_sharding_context &&
      config.logical_sharding_context->loop_submesh.has_value()) {
    return config.logical_sharding_context->loop_submesh->SubmeshForIteration(
        0);
  }
  return config.devices;
}

int64_t HloPartition::NumDevicesForInstruction(
    const HloInstruction* instruction) const {
  auto color = Color(instruction);
  if (color.has_value()) {
    const auto& config = ConfigForColor(*color);
    if (config.logical_sharding_context &&
        config.logical_sharding_context->loop_submesh.has_value()) {
      return config.logical_sharding_context->loop_submesh->TaskMeshSize();
    }
    return config.devices.size();
  }
  return devices_.size();
}

bool HloPartition::EquivalentMesh(const std::string& lhs_color,
                                  const std::string& rhs_color) const {
  const auto& lhs_config = ConfigForColor(lhs_color);
  const auto& rhs_config = ConfigForColor(rhs_color);

  if (lhs_config.devices.size() != rhs_config.devices.size()) {
    return false;
  }

  if (lhs_config.logical_sharding_context &&
      rhs_config.logical_sharding_context) {
    if (lhs_config.logical_sharding_context->loop_submesh.has_value() &&
        rhs_config.logical_sharding_context->loop_submesh.has_value()) {
      return lhs_config.logical_sharding_context->loop_submesh
                 ->TaskMeshSize() ==
             rhs_config.logical_sharding_context->loop_submesh->TaskMeshSize();
    } else if (lhs_config.logical_sharding_context->loop_submesh.has_value() ||
               rhs_config.logical_sharding_context->loop_submesh.has_value()) {
      // one has a loop submesh, the other does not
      return false;
    }
  } else if (lhs_config.logical_sharding_context &&
             lhs_config.logical_sharding_context->loop_submesh.has_value()) {
    // one has a loop submesh, the other does not
    return false;
  } else if (rhs_config.logical_sharding_context &&
             rhs_config.logical_sharding_context->loop_submesh.has_value()) {
    return false;
  }
  // devices match and neither has a loop submesh
  return true;
}

bool HloPartition::SameMesh(const HloInstruction* lhs,
                            const HloInstruction* rhs) const {
  auto lhs_color = Color(lhs);
  if (!lhs_color.has_value()) {
    VLOG(5) << lhs->name() << " does not have a color";
    return false;
  }

  auto rhs_color = Color(rhs);
  if (!rhs_color.has_value()) {
    VLOG(5) << rhs->name() << " does not have a color";
    return false;
  }

  const auto& lhs_config = ConfigForColor(*lhs_color);
  const auto& rhs_config = ConfigForColor(*rhs_color);

  if (lhs_config.devices != rhs_config.devices) {
    VLOG(5) << lhs->name() << " and " << rhs->name()
            << " have different devices: " << lhs_config.devices
            << " != " << rhs_config.devices;
    return false;
  }

  if (lhs_config.logical_sharding_context &&
      rhs_config.logical_sharding_context) {
    if (lhs_config.logical_sharding_context->loop_submesh.has_value() &&
        rhs_config.logical_sharding_context->loop_submesh.has_value()) {
      return lhs_config.logical_sharding_context->loop_submesh ==
             rhs_config.logical_sharding_context->loop_submesh;
    } else if (lhs_config.logical_sharding_context->loop_submesh.has_value() ||
               rhs_config.logical_sharding_context->loop_submesh.has_value()) {
      // one has a loop submesh, the other does not
      return false;
    }
  } else if (lhs_config.logical_sharding_context ||
             rhs_config.logical_sharding_context) {
    // one has an autoshard context, the other does not
    return false;
  }
  // devices match and neither has a loop submesh
  return true;
}

absl::StatusOr<HloPartition> HloPartition::Create(HloModule* module,
                                                  zuku::DeviceList devices) {
  return HloPartition{module, devices};
}

absl::StatusOr<std::string> HloPartition::FindOrAllocateColor(
    zuku::DeviceList devices, std::shared_ptr<LogicalShardingContext> context) {
  // we have to match the name
  auto iter = device_list_to_colors_.find(devices);
  if (iter == device_list_to_colors_.end()) {
    return AllocateColor(
        absl::StrCat("devices_", devices.start(), "-", devices.stop()), devices,
        std::move(context));
  }
  if (iter->second.empty()) {
    return InternalStrCat(
        "device list added to map, but has no assigned colors");
  }
  return iter->second.front();
}

absl::StatusOr<std::string> HloPartition::FindOrAllocateColor(
    std::string name, zuku::DeviceList devices,
    std::shared_ptr<LogicalShardingContext> context) {
  // we have to match the name
  auto iter = colors_.find(name);
  if (iter == colors_.end()) {
    return AllocateColor(std::move(name), devices, std::move(context));
  }

  if (iter->second.devices != devices) {
    LOG(ERROR) << devices << " != " << iter->second.devices;
    return InvalidArgumentStrCat("color ", name,
                                 " allocated again with different devices");
  }

  return std::move(name);
}

absl::StatusOr<std::string> HloPartition::FindOrAllocateGlobalColor() {
  // no autosharding on the global context for now
  return FindOrAllocateColor(devices_, nullptr);
}

absl::StatusOr<std::string> HloPartition::FindOrAllocateGlobalColor(
    std::string name) {
  // no autosharding on the global context for now
  return FindOrAllocateColor(std::move(name), devices_, nullptr);
}

absl::StatusOr<std::string> HloPartition::AllocateLoopIncrementColor() {
  return FindOrAllocateGlobalColor(std::string(kLoopIncrementColor));
}

bool HloPartition::IsLoopIncrementColor(const std::string& color) const {
  return absl::StartsWith(color, kLoopIncrementColor);
}

absl::StatusOr<std::string> HloPartition::FindOrAllocateDefaultColor(
    std::optional<std::string> name) {
  if (name.has_value()) {
    return FindOrAllocateColor(*std::move(name), devices_, nullptr);
  }

  // see if we have a color allocated across the default device list
  auto iter = device_list_to_colors_.find(devices_);
  if (iter != device_list_to_colors_.end()) {
    return iter->second.front();
  }

  // no existing allocated colors on the default devices, make one now
  return FindOrAllocateColor(
      absl::StrCat("devices_", devices_.start(), "...", devices_.stop()),
      devices_, nullptr);
}

bool HloPartition::HasColor(const std::string& color) const {
  return colors_.contains(color);
}

std::optional<std::string> HloPartition::FindColor(
    const zuku::DeviceList& devices) const {
  auto iter = device_list_to_colors_.find(devices);
  if (iter == device_list_to_colors_.end()) {
    return std::nullopt;
  }
  return iter->second.front();
}

absl::StatusOr<std::string> HloPartition::AllocateColor(
    std::string name, zuku::DeviceList devices,
    std::shared_ptr<LogicalShardingContext> context) {
  const int64_t color = max_color_++;

  if (colors_.contains(name)) {
    // uniquify the color name
    absl::StrAppend(&name, ".", color);
  }

  VLOG(5) << this << " allocating color " << color << " for " << name
          << " on devices " << devices << ", context=" << context;

  auto& config = colors_[name];
  if (context) {
    config.logical_sharding_context = std::move(context);
  }

  device_list_to_colors_[devices].push_back(name);
  config.name = name;
  config.devices = std::move(devices);
  return name;
}

const HloPartition::ColorConfig& HloPartition::ConfigForColor(
    const std::string& color) const {
  auto iter = colors_.find(color);
  if (iter == colors_.end()) {
    LOG(FATAL) << "no config exists for color " << color;
  }
  return iter->second;
}

const zuku::DeviceList& HloPartition::DevicesForColor(
    const std::string& color) const {
  return ConfigForColor(color).devices;
}

const zuku::DeviceList& HloPartition::DevicesForInstruction(
    HloInstruction* instruction) const {
  auto color = Color(instruction);
  if (color.has_value()) {
    return DevicesForColor(*color);
  }
  return devices_;
}

absl::StatusOr<std::optional<std::string>>
HloPartition::ComputeMetadataNameColor(HloInstruction* instruction) {
  if (!enable_metadata_name_tasks) {
    return std::nullopt;
  }
  const std::string& metadata_op_name = instruction->metadata().op_name();
  if (metadata_op_name.empty()) {
    VLOG(5) << "no metadata for " << instruction->name()
            << ", cannot compute implicit color";
    return std::nullopt;
  }
  VLOG(5) << "computing implicit color for " << instruction->name();
  for (const auto& context : NamedTaskContexts()) {
    std::string matched_name;
    VLOG(5) << instruction->name() << " trying to match "
            << context.matcher->pattern() << " against " << metadata_op_name;
    if (re2::RE2::PartialMatch(metadata_op_name, *context.matcher,
                               &matched_name)) {
      if (matched_name.empty()) {
        return InvalidArgumentStrCat("matcher '", context.matcher->pattern(),
                                     " produced an empty match on ",
                                     metadata_op_name);
      }

      const bool backprop =
          absl::StrContains(metadata_op_name, kJvpTransposeMetadata);
      auto match_iter = matched_colors_.find({matched_name, backprop});
      if (match_iter != matched_colors_.end()) {
        return match_iter->second;
      }

      auto [devices, color_name] = context.callback(matched_name, backprop);
      const int64_t start = devices.first;
      const int64_t num_devices = devices.second - start;
      zuku::DeviceList dl{{.start = start, .num_devices = num_devices}};
      auto config_iter = colors_.find(color_name);
      if (config_iter != colors_.end()) {
        if (dl != config_iter->second.devices) {
          return InvalidArgumentStrCat(
              "task ", matched_name, " on color ", color_name,
              " was previously allocated with different device list, cannot "
              "have multiple submeshes for a color");
        }
        return color_name;
      }

      TF_ASSIGN_OR_RETURN(auto context,
                          MakeLogicalShardingContext(dl, context));
      TF_ASSIGN_OR_RETURN(
          const std::string uniquified_name,
          AllocateColor(color_name, std::move(dl), std::move(context)));
      if (uniquified_name != color_name) {
        return InternalStrCat(
            "color ", color_name,
            " was already allocated with mismatched config, cannot reallocate");
      }

      VLOG(5) << "Adding new metadata name task for color=" << color_name
              << " on context=" << matched_name;
      matched_colors_[{matched_name, backprop}] = color_name;
      return color_name;
    }
  }
  return std::nullopt;
}

bool ContainsLegateCustomCall(const HloModuleProto& proto) {
  for (const HloComputationProto& comp : proto.computations()) {
    for (const HloInstructionProto& instr : comp.instructions()) {
      if (instr.opcode() == "custom-call" &&
          kLegateCustomCallTargets.contains(instr.custom_call_target())) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace xla

extern "C" void SetEnableMetadataNameTasks(bool flag) {
  xla::enable_metadata_name_tasks = flag;
}

extern "C" void RegisterMetadataNameTask(
    std::string matcher, xla::HloPartition::name_and_devices_fxn callback,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes) {
  // TODO, add an API for this
  std::optional<int64_t> loop_submesh_size;
  bool loop_submesh_reverse = false;

  VLOG(5) << "Registering implicit task " << matcher;

  if (dims.size() != axes.size()) {
    throw std::invalid_argument(
        absl::StrCat("no. dims does not match no. axes (", dims.size(),
                     " != ", axes.size(), ")"));
  }

  xla::NamedTaskContext task{.matcher = std::make_unique<re2::RE2>(matcher),
                             .callback = std::move(callback),
                             .dims = std::move(dims),
                             .device_axes = std::move(axes),
                             .logical_axes = std::move(logical_axes)};

  xla::NamedTaskContexts().push_back(std::move(task));
}

extern "C" void EnableLegateRecomputation(bool enable) {
  xla::enable_recomputation = enable;
}

extern "C" void ClearMetadataNameTasks() { xla::NamedTaskContexts().clear(); }
