/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_INSTRUCTION_H_
#define XLA_PJRT_MULTIMESH_MPMD_INSTRUCTION_H_

#include <optional>
#include <type_traits>

#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

using LogicalShardingAxes =
    absl::InlinedVector<absl::InlinedVector<std::string, 6>, 6>;

// If the `source` instruction has been assigned a color and the
// `target` instruction has not been assigned a color, assign
// the source partition color to the target.
bool PropagateColor(const HloInstruction* source, HloInstruction* target);

// If the `source` instruction has been assigned logical sharding axes and the
// `target` instruction has not been assigned target axes, assign
// the source axes to the target.
bool PropagateAxes(const HloInstruction* source, HloInstruction* target);

// Propagate all assigned properties (parition color, logical axes)
// from the `source` instruction to the `target` instruction.
void PropagateProperties(const HloInstruction* source, HloInstruction* target);

// Assign the given `color` to `instruction`, overwriting
// any existing color assignment.
void AssignColor(HloInstruction* instruction, std::string color);

// Removes any existing partition color from the given `instruction`.
void RemoveColor(HloInstruction* instruction);

// Assigns a uniform color to the given `instruction` if all of the
// operands have been assigned a color and that color is the same.
void ColorTuple(HloInstruction* instruction);

// Returns whether the given `instruction` has been assigned partition color.
bool IsAssignedColor(const HloInstruction* instruction);

// If `instruction` has been assigned a partition color, returns the assigned
// color. If no color has been assigned, returns std::nullopt.
std::optional<std::string> Color(const HloInstruction* instruction);

// If the `instruction` has been assigned a partition color, returns the
// assigned color. Otherwise, the function returns a default partition color
// name.
std::string ColorOrDefault(const HloInstruction* instruction);

// If the `instruction` has been assigned logical axes, return the axes.
// Otherwise returns std::nullopt.
std::optional<LogicalShardingAxes> GetAxes(const HloInstruction* instruction);

// Returns whether the given `instruction` has been assigned logical axes.
bool HasAssignedAxes(const HloInstruction* instruction);

// Assigns logical `axes` encoded as a Json string to the given `instruction`.
// This overwrites any existing axis assignments.
void AssignAxes(HloInstruction* instruction, std::string axes);

// If no axes have been assigned to the given `instruction`,
// assigns the logical `axes`, encoded as a JSON string.
void AssignAxesIfUnassigned(HloInstruction* instruction, std::string axes);

// Assigns logical `axes` to the given `instruction`. This overwrites
// any existing axis assignments.
void AssignAxes(HloInstruction* instruction, const LogicalShardingAxes& axes);

// For the given `instruction`, if logical axes have been assigned, this
// returns the logical axes encoded as a JSON string. If no axes have
// been assigned, return std::nullopt.
std::optional<std::string> GetAxesString(const HloInstruction* instruction);

void ApplyRootTupleShardingsToOperands(HloModule* module, HloInstruction* root);

void AddAttribute(HloInstruction* instruction, std::string name,
                  std::string value);

absl::StatusOr<std::string> GetAttribute(HloInstruction* instruction,
                                         const std::string& name);

template <class T>
absl::StatusOr<T> GetAttribute(HloInstruction* instruction,
                               const std::string& name) {
  TF_ASSIGN_OR_RETURN(auto value, GetAttribute(instruction, name));
  if constexpr (std::is_integral_v<T>) {
    T t;
    bool valid = absl::SimpleAtoi(value, &t);
    if (!valid) {
      return InvalidArgumentStrCat("attribute ", value, " for ", name,
                                   " is not a valid integer");
    }
    return t;
  } else if constexpr (std::is_floating_point_v<T>) {
    T t;
    bool valid = absl::SimpleAtof(value, &t);
    if (!valid) {
      return InvalidArgumentStrCat("attribute ", value, " for ", name,
                                   " is not a valid floating point");
    }
    return t;
  }
  return InvalidArgumentStrCat(
      "neither integral nor floating point type specified for attribute ",
      value, " for ", name);
}

template <class T>
void AddAttribute(HloInstruction* instruction, std::string name, const T& t) {
  AddAttribute(instruction, std::move(name), absl::StrCat(t));
}

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_INSTRUCTION_H_
