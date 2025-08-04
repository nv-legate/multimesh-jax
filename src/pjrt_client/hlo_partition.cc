/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/hlo_partition.h"

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
#include "xla/pjrt/multimesh/json_utils.h"
#include "xla/pjrt/multimesh/mm_sharding.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/pjrt/multimesh/pycallback_types.h"
#include "xla/util.h"

namespace xla {

namespace {

bool enable_metadata_name_tasks = true;
bool enable_recomputation = false;

constexpr absl::string_view kLoopIncrementColor = "loop_increment";
constexpr absl::string_view kJvpTransposeMetadata = "transpose(jvp";
constexpr char kMultiMeshTaskCustomCallTarget[] = "MultiMeshTask";
constexpr char kMicrobatchCustomCallTarget[] = "Microbatch";
constexpr char kMicrobatchInitCustomCallTarget[] = "MicrobatchInit";
constexpr char kMicrobatchSliceCustomCallTarget[] = "MicrobatchSlice";
constexpr char kAutoShardingCustomCallTarget[] = "AutoSharding";

const absl::flat_hash_set<std::string> kMultiMeshCustomCallTargets = {
    kMultiMeshTaskCustomCallTarget, kMicrobatchCustomCallTarget,
    kMicrobatchInitCustomCallTarget, kMicrobatchSliceCustomCallTarget,
    kAutoShardingCustomCallTarget};

}  // namespace

absl::StatusOr<Json::Value> GetJson(HloInstruction* instruction) {
  auto json = GetJsonValue(instruction->raw_backend_config_string().data(),
                           instruction->raw_backend_config_string().size());
  if (!json.ok()) {
    return InvalidArgumentStrCat("could not parse backend config for: ",
                                 json.status().message());
  }
  return json;
}

absl::StatusOr<Json::Value> GetJson(const HloInstructionProto& instr) {
  auto json = GetJsonValue(instr.backend_config().data(),
                           instr.backend_config().size());
  if (!json.ok()) {
    return InvalidArgumentStrCat("could not parse backend config for ",
                                 instr.name(), ": ", json.status().message());
  }
  return json;
}

// Struct holding all the metadata for matching implicit tasks
// and mapping matched names to physical submeshes
struct NamedTask {
  std::unique_ptr<re2::RE2> matcher;
  std::function<multimesh::TaskOptions(const std::string&, bool)> callback;
};

class NamedTaskContextStack {
 public:
  // enqueue a single empty vector so there is always one stack
  NamedTaskContextStack() { tasks_.emplace_back(); }

  auto begin() const { return tasks_.rbegin(); }

  auto end() const { return tasks_.rend(); }

  void clear() {
    tasks_.clear();
    tasks_.emplace_back();
  }

  void push() { tasks_.emplace_back(); }

  void pop() { tasks_.pop_back(); }

  void push_back(NamedTask task) { tasks_.back().push_back(std::move(task)); }

 private:
  std::vector<std::vector<NamedTask>> tasks_;
};

// Static helper for returning implicit task map mapping
// matched names to their task metadata
NamedTaskContextStack& NamedTaskContexts() {
  static auto* named_task_contexts = new NamedTaskContextStack;
  return *named_task_contexts;
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
    {
      auto iter = original_color_.find(old_color);
      if (iter != original_color_.end()) {
        // absl pointer stability
        std::string mapped_color = iter->second;
        original_color_[new_color] = std::move(mapped_color);
      } else {
        original_color_[new_color] = old_color;
      }
    }
  }
  colors_ = std::move(new_configs);
  return absl::OkStatus();
}

const multimesh::TaskOptions& HloPartition::GetTaskOptions(
    const std::string& color) const {
  return ConfigForColor(color).task_options;
}

zuku::DeviceList HloPartition::DefaultPerTaskMesh(
    const std::string& color) const {
  const auto& config = ConfigForColor(color);
  return config.devices;
}

int64_t HloPartition::NumDevicesForInstruction(
    const HloInstruction* instruction) const {
  auto color = Color(instruction);
  if (color.has_value()) {
    const auto& config = ConfigForColor(*color);
    return config.devices.size();
  }
  return devices_.size();
}

bool HloPartition::IsBackprop(const std::string& color) const {
  auto iter = colors_.find(color);
  if (iter == colors_.end()) {
    return false;
  }
  return iter->second.backprop;
}

bool HloPartition::IsBackprop(const HloInstruction* instruction) const {
  auto color = Color(instruction);
  if (color.has_value()) {
    return !IsBackprop(*color);
  }
  return false;
}

bool HloPartition::IsNonCritical(const std::string& color) const {
  auto iter = colors_.find(color);
  if (iter == colors_.end()) {
    return false;
  }
  return iter->second.noncritical;
}

bool HloPartition::IsCritical(const HloInstruction* instruction) const {
  return !IsNonCritical(*Color(instruction));
}

bool HloPartition::IsNonCritical(const HloInstruction* instruction) const {
  auto color = Color(instruction);
  if (color.has_value()) {
    return IsNonCritical(*color);
  }
  return false;
}

void HloPartition::SetColorAsNonCritical(const std::string& color) {
  colors_[color].noncritical = true;
}

bool HloPartition::EquivalentMesh(const std::string& lhs_color,
                                  const std::string& rhs_color) const {
  const auto& lhs_config = ConfigForColor(lhs_color);
  const auto& rhs_config = ConfigForColor(rhs_color);

  if (lhs_config.devices.size() != rhs_config.devices.size()) {
    return false;
  }

  if (lhs_config.task_options.loop_dependent_devices == nullptr &&
      rhs_config.task_options.loop_dependent_devices == nullptr) {
    return true;
  }

  // if loop dependent meshes, then we have to say these are different meshes
  return false;
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

  if (lhs_config.task_options.loop_dependent_devices == nullptr &&
      rhs_config.task_options.loop_dependent_devices == nullptr) {
    return true;
  }
  return false;
}

absl::StatusOr<HloPartition> HloPartition::Create(HloModule* module,
                                                  zuku::DeviceList devices) {
  return HloPartition{module, devices};
}

absl::StatusOr<std::string> HloPartition::FindOrAllocateDefaultColor(
    zuku::DeviceList devices) {
  // we have to match the name
  auto iter = device_list_to_colors_.find(devices);
  if (iter == device_list_to_colors_.end()) {
    return AllocateColor(
        absl::StrCat("devices_", devices.start(), "-", devices.stop()), devices,
        {});
  }
  if (iter->second.empty()) {
    return InternalStrCat(
        "device list added to map, but has no assigned colors");
  }
  return iter->second.front();
}

absl::StatusOr<std::string> HloPartition::FindOrAllocateGlobalColor() {
  // no autosharding on the global context for now
  return FindOrAllocateDefaultColor(devices_);
}

absl::StatusOr<std::string> HloPartition::AllocateLoopIncrementColor() {
  return AllocateColor(std::string(kLoopIncrementColor), devices_, {});
}

bool HloPartition::IsLoopIncrementColor(const std::string& color) const {
  return absl::StartsWith(color, kLoopIncrementColor);
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

absl::StatusOr<std::string> HloPartition::CloneColor(
    const std::string& existing_color, std::string new_color,
    std::optional<zuku::DeviceList> devices_override,
    std::optional<multimesh::TaskOptions> task_options_override) {
  if (!colors_.contains(existing_color)) {
    return InvalidArgumentStrCat(
        "attempting to clone from non-existent color: ", existing_color);
  }
  if (colors_.contains(new_color)) {
    return InvalidArgumentStrCat("color ", new_color, "exists when cloning ",
                                 existing_color);
  }
  ColorConfig new_config = colors_.at(existing_color);

  // override attributes if given
  if (devices_override.has_value()) {
    new_config.devices = *devices_override;
  }
  if (task_options_override.has_value()) {
    new_config.task_options = *task_options_override;
  }

  colors_[new_color] = std::move(new_config);
  return new_color;
}

absl::StatusOr<std::string> HloPartition::AllocateColor(ColorConfig config) {
  const int64_t color = max_color_++;

  if (colors_.contains(config.name)) {
    // uniquify the color name
    absl::StrAppend(&config.name, ".", color);
  }

  VLOG(5) << this << " allocating color " << color << " for " << config.name
          << " on devices " << config.devices;

  device_list_to_colors_[config.devices].push_back(config.name);
  std::string copy = config.name;
  colors_[config.name] = std::move(config);
  return copy;
}

absl::StatusOr<std::string> HloPartition::AllocateColor(
    absl::string_view name, zuku::DeviceList devices,
    multimesh::TaskOptions task_options) {
  return AllocateColor({.name = std::string(name),
                        .devices = std::move(devices),
                        .task_options = std::move(task_options)});
}

std::string HloPartition::OriginalColor(const std::string& color) const {
  auto iter = original_color_.find(color);
  if (iter != original_color_.end()) {
    return iter->second;
  }
  return color;
}

const HloPartition::ColorConfig& HloPartition::ConfigForColor(
    const std::string& color) const {
  auto iter = colors_.find(color);
  if (iter == colors_.end()) {
    LOG(FATAL) << "no config exists for color " << color;
  }
  return iter->second;
}

const HloPartition::ColorConfig& HloPartition::ConfigForInstruction(
    const HloInstruction* instruction) const {
  return ConfigForColor(*Color(instruction));
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
  // Loop through the context stack. Prioritize the most recent contexts pushed,
  // but look for matches in all outer contexts if none found.
  for (const auto& context : NamedTaskContexts()) {
    for (const auto& task : context) {
      std::string matched_name;
      VLOG(5) << instruction->name() << " trying to match "
              << task.matcher->pattern() << " against " << metadata_op_name;
      if (re2::RE2::PartialMatch(metadata_op_name, *task.matcher,
                                 &matched_name)) {
        if (matched_name.empty()) {
          return InvalidArgumentStrCat("matcher '", task.matcher->pattern(),
                                       " produced an empty match on ",
                                       metadata_op_name);
        }

        const bool backprop =
            absl::StrContains(metadata_op_name, kJvpTransposeMetadata);
        auto match_iter = matched_colors_.find({matched_name, backprop});
        if (match_iter != matched_colors_.end()) {
          return match_iter->second;
        }

        auto task_options = task.callback(matched_name, backprop);
        if (!task_options.name.has_value()) {
          return InvalidArgumentStrCat(
              "matcher ", matched_name,
              " produed TaskOptions with no name specified");
        }

        if (task_options.devices.empty()) {
          return InvalidArgumentStrCat(matched_name, " with color ",
                                       *task_options.name,
                                       " returned empty device vector");
        }

        std::string name = *std::move(task_options.name);
        const int64_t start = task_options.devices.front();
        const int64_t num_devices = task_options.devices.size();
        zuku::DeviceList dl{{.start = start, .num_devices = num_devices}};
        auto config_iter = colors_.find(name);
        if (config_iter != colors_.end()) {
          if (dl != config_iter->second.devices) {
            return InvalidArgumentStrCat(
                "task ", matched_name, " on color ", name,
                " was previously allocated with different device list, cannot "
                "have multiple submeshes for a color");
          }
          return name;
        }

        TF_ASSIGN_OR_RETURN(
            const std::string uniquified_name,
            AllocateColor(name, std::move(dl), std::move(task_options)));
        if (uniquified_name != name) {
          return InternalStrCat("color ", name,
                                " was already allocated with mismatched "
                                "config, cannot reallocate");
        }

        colors_[name].backprop = backprop;

        VLOG(5) << "Adding new metadata name task for color=" << name
                << " on context=" << matched_name;
        matched_colors_[{matched_name, backprop}] = name;
        return name;
      }
    }
  }
  return std::nullopt;
}

bool ContainsMultiMeshCustomCall(const HloModuleProto& proto) {
  for (const HloComputationProto& comp : proto.computations()) {
    for (const HloInstructionProto& instr : comp.instructions()) {
      if (instr.opcode() == "custom-call" &&
          kMultiMeshCustomCallTargets.contains(instr.custom_call_target())) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace xla

extern "C" void RegisterMetadataNameTask(
    std::string matcher,
    std::function<multimesh::TaskOptions(const std::string&, bool)> callback) {
  VLOG(5) << "Registering implicit task " << matcher;

  xla::NamedTask task{.matcher = std::make_unique<re2::RE2>(matcher),
                      .callback = std::move(callback)};

  xla::NamedTaskContexts().push_back(std::move(task));
}

extern "C" void EnableMultiMeshRecomputation(bool enable) {
  xla::enable_recomputation = enable;
}

extern "C" void SetEnableMetadataNameTasks(bool flag) {
  xla::enable_metadata_name_tasks = flag;
}

extern "C" void PopMetadataNameTaskContext() { xla::NamedTaskContexts().pop(); }

extern "C" void PushMetadataNameTaskContext() {
  xla::NamedTaskContexts().push();
}

extern "C" void ClearMetadataNameTasks() { xla::NamedTaskContexts().clear(); }
