/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_instruction.h"

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/multimesh/json_utils.h"
#include "xla/pjrt/multimesh/logical_sharding_context.h"

namespace xla {
namespace {

constexpr absl::string_view kDefaultColorName = "unassigned";

}

void ApplyRootTupleShardingsToOperands(HloModule* module,
                                       HloInstruction* root) {
  if (root->has_sharding() && root->shape().IsTuple()) {
    auto get_sharding = [&](int64_t index) {
      if (root->sharding().IsTuple()) {
        return root->sharding().tuple_elements()[index];
      }
      return root->sharding();
    };
    for (int64_t index = 0; index < root->operand_count(); ++index) {
      const HloSharding& sharding = get_sharding(index);
      const bool allow_sharding_overwrite = [&] {
        if (module->config()
                .allow_spmd_sharding_propagation_to_output()
                .size() == 1) {
          return module->config()
                     .allow_spmd_sharding_propagation_to_output()[0] &&
                 sharding.IsReplicated();
        } else if (module->config()
                       .allow_spmd_sharding_propagation_to_output()
                       .size() > index) {
          return module->config()
                     .allow_spmd_sharding_propagation_to_output()[index] &&
                 sharding.IsReplicated();
        }
        return false;
      }();

      auto* operand = root->mutable_operand(index);
      if (!allow_sharding_overwrite && !operand->has_sharding()) {
        operand->set_sharding(get_sharding(index));
      }
    }
  }
}

void AddAttribute(HloInstruction* instruction, std::string name,
                  std::string value) {
  FrontendAttributes attrs = instruction->frontend_attributes();
  (*attrs.mutable_map())[std::move(name)] = std::move(value);
  instruction->set_frontend_attributes(std::move(attrs));
}

absl::StatusOr<std::string> GetAttribute(HloInstruction* instruction,
                                         const std::string& name) {
  if (instruction->frontend_attributes().map().contains(name)) {
    return instruction->frontend_attributes().map().at(name);
  }
  return InvalidArgumentStrCat("attribute ", name, " not found on instruction");
}

void AssignColor(HloInstruction* instruction, std::string color) {
  VLOG(5) << "assigning color " << color << " to " << instruction->name();
  // add frontend attributes does not overwrite the existing ones
  FrontendAttributes attrs = instruction->frontend_attributes();
  (*attrs.mutable_map())["color"] = std::move(color);
  instruction->set_frontend_attributes(std::move(attrs));
}

bool PropagateColor(const HloInstruction* source, HloInstruction* target) {
  if (source->frontend_attributes().map().contains("color")) {
    AssignColor(target, source->frontend_attributes().map().at("color"));
    return true;
  }
  return false;
}

std::optional<std::string> Color(const HloInstruction* instruction) {
  if (instruction->frontend_attributes().map().contains("color")) {
    return instruction->frontend_attributes().map().at("color");
  }
  return std::nullopt;
}

std::string ColorOrDefault(const HloInstruction* instruction) {
  auto color = Color(instruction);
  if (color.has_value()) {
    return *std::move(color);
  }
  return std::string(kDefaultColorName);
}

void ColorTuple(HloInstruction* instruction) {
  std::optional<std::string> uniform_color{std::nullopt};
  for (auto* operand : instruction->operands()) {
    auto operand_color = Color(operand);
    if (operand_color.has_value()) {
      if (!uniform_color.has_value()) {
        uniform_color = Color(operand);
      } else if (*uniform_color != *operand_color) {
        uniform_color = std::nullopt;
        break;
      }
    } else {
      uniform_color = std::nullopt;
      break;
    }
  }

  // tuples should only be assigned a uniform color when ALL of the
  // operands have been assigned a color and that color is the same
  if (uniform_color.has_value()) {
    VLOG(5) << "assigning uniform color " << *uniform_color << " to tuple "
            << instruction->name();
    AssignColor(instruction, *std::move(uniform_color));
  }

  return absl::OkStatus();
}

bool IsAssignedColor(const HloInstruction* instruction) {
  return instruction->frontend_attributes().map().contains("color");
}

void RemoveColor(HloInstruction* instruction) {
  FrontendAttributes attrs = instruction->frontend_attributes();
  attrs.mutable_map()->erase("color");
  instruction->set_frontend_attributes(std::move(attrs));
}

bool PropagateAxes(const HloInstruction* source, HloInstruction* target) {
  const auto& src_map = source->frontend_attributes().map();
  auto iter = src_map.find("axes");
  if (iter != src_map.end()) {
    AssignAxesIfUnassigned(target, iter->second);
    return true;
  }
  return false;
}

void PropagateProperties(const HloInstruction* source, HloInstruction* target) {
  PropagateColor(source, target);
  PropagateAxes(source, target);
}

void AssignAxes(HloInstruction* instruction, std::string axes) {
  FrontendAttributes attrs{};
  (*attrs.mutable_map())["axes"] = std::move(axes);
  instruction->add_frontend_attributes(std::move(attrs));
}

void AssignAxesIfUnassigned(HloInstruction* instruction, std::string axes) {
  // this doesn't overwrite an existing attribute
  FrontendAttributes attrs{};
  (*attrs.mutable_map())["axes"] = std::move(axes);
  instruction->add_frontend_attributes(std::move(attrs));
}

bool HasAssignedAxes(const HloInstruction* instruction) {
  return instruction->frontend_attributes().map().contains("axes");
}

std::optional<std::string> GetAxesString(const HloInstruction* instruction) {
  const auto& src_map = instruction->frontend_attributes().map();
  auto iter = src_map.find("axes");
  if (iter != src_map.end()) {
    return iter->second;
  }
  return std::nullopt;
}

std::optional<LogicalShardingAxes> GetAxes(const HloInstruction* instruction) {
  const auto& src_map = instruction->frontend_attributes().map();
  auto iter = src_map.find("axes");
  if (iter != src_map.end()) {
    auto json = GetJsonValue(iter->second.data(), iter->second.size());
    auto axes = GetTaskValue<InlineVector<InlineVector<std::string>>>(
        *json, std::string(instruction->name()), "axes");

    return LogicalShardingAxes{.axes = *std::move(axes)};
  }
  return std::nullopt;
}

void AssignAxes(HloInstruction* instruction, const LogicalShardingAxes& axes) {
  Json::FastWriter writer;
  Json::Value json_axes;
  // expected to be a field 'axes' in a json dict
  json_axes["axes"] = ToJson(axes.axes);
  // ugh, can't print without the new line
  auto json_string = writer.write(json_axes);
  json_string = json_string.substr(0, json_string.size() - 1);
  AssignAxes(instruction, std::move(json_string));
}

}  // namespace xla
