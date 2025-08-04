/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/task_splitter.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <queue>

#include "mpmd_utils.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/util.h"

namespace xla {
namespace {

struct TaskConfig {
  std::vector<HloInstruction*> external_parameters;
  std::vector<HloInstruction*> internal_parameters;
  std::vector<HloInstruction*> external_roots;
  std::vector<HloInstruction*> internal_roots;
};

}  // namespace

absl::StatusOr<std::vector<std::string>> SplitTask(
    HloInstruction* task,
    const std::vector<std::vector<HloInstruction*>>& instructions,
    const std::vector<std::string>& suffixes, HloPartition* partition) {
  HloComputation* parent = task->parent();
  HloCloneContext context{parent->parent()};
  HloComputation* computation = task->called_computations()[0];
  std::vector<TaskConfig> configs;
  for (const auto& task_instruction_set : instructions) {
    TaskConfig config;
    absl::flat_hash_set<HloInstruction*> included{task_instruction_set.begin(),
                                                  task_instruction_set.end()};
    absl::flat_hash_set<HloInstruction*> operands_visited;
    for (auto* instruction : task_instruction_set) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        // don't visit directly
        continue;
      }
      for (auto* operand : instruction->mutable_operands()) {
        if (operands_visited.contains(operand)) {
          continue;
        }
        operands_visited.insert(operand);
        if (operand->opcode() == HloOpcode::kParameter) {
          config.external_parameters.push_back(operand);
        } else if (!included.contains(operand)) {
          config.internal_parameters.push_back(operand);
        }
      }
      for (auto* user : instruction->users()) {
        if (user == computation->root_instruction()) {
          config.external_roots.push_back(instruction);
          break;
        } else if (!included.contains(user)) {
          config.internal_roots.push_back(instruction);
          break;
        }
      }
    }
    configs.push_back(std::move(config));
  }

  std::string color = *Color(task);

  std::vector<std::string> new_colors;
  std::vector<HloInstruction*> new_computations;
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      external_replacements;
  for (int64_t task_number = 0; task_number < instructions.size();
       ++task_number) {
    HloComputation::Builder builder{
        absl::StrCat(computation->name(), suffixes[task_number])};
    TaskConfig& config = configs[task_number];
    const std::vector<HloInstruction*>& task_instructions =
        instructions[task_number];

    absl::flat_hash_map<const HloInstruction*, HloInstruction*> clone_map;

    int64_t param_number = 0;
    auto clone_param = [&](HloInstruction* param) {
      auto new_param = HloInstruction::CreateParameter(
          param_number++, param->shape(), param->name());
      new_param->set_sharding(param->sharding_ptr());
      return new_param;
    };

    TF_ASSIGN_OR_RETURN(std::string new_color,
                        partition->CloneColor(
                            color, absl::StrCat(color, suffixes[task_number])));

    new_colors.push_back(new_color);

    std::vector<HloInstruction*> new_call_operands;
    for (HloInstruction* param : config.external_parameters) {
      TF_ASSIGN_OR_RETURN(clone_map[param],
                          builder.AddParameter(clone_param(param)));
      new_call_operands.push_back(
          task->mutable_operand(param->parameter_number()));
    }
    for (HloInstruction* param : config.internal_parameters) {
      TF_ASSIGN_OR_RETURN(clone_map[param],
                          builder.AddParameter(clone_param(param)));
      new_call_operands.push_back(external_replacements.at(param));
    }

    for (auto* instruction : task_instructions) {
      if (instruction->opcode() == HloOpcode::kParameter ||
          instruction->IsRoot()) {
        // cloned parameter
        continue;
      }

      absl::InlinedVector<HloInstruction*, 2> new_operands;
      for (auto* operand : instruction->operands()) {
        new_operands.push_back(clone_map.at(operand));
      }
      clone_map[instruction] =
          builder.AddInstruction(instruction->CloneWithNewOperands(
              instruction->shape(), std::move(new_operands), &context));
    }

    absl::InlinedVector<HloInstruction*, 2> new_root_operands;
    new_root_operands.reserve(config.external_roots.size() +
                              config.internal_roots.size());
    for (auto* instruction : config.external_roots) {
      new_root_operands.push_back(clone_map.at(instruction));
    }
    for (auto* instruction : config.internal_roots) {
      new_root_operands.push_back(clone_map.at(instruction));
    }

    auto* root =
        builder.AddInstruction(HloInstruction::CreateTuple(new_root_operands));
    HloComputation* split_computation =
        computation->parent()->AddComputationAndUnifyNamesAndIds(
            builder.Build(root), /*is_entry=*/false);
    for (auto* subinstr : split_computation->instructions()) {
      AssignColor(subinstr, new_color);
    }

    HloInstruction* subtask = parent->AddInstruction(HloInstruction::CreateCall(
        root->shape(), new_call_operands, split_computation));
    AssignColor(subtask, new_color);

    auto clone_root = [&](HloInstruction* instruction, int64_t index) {
      HloInstruction* clone = parent->AddInstruction(
          HloInstruction::CreateGetTupleElement(subtask, index));
      AssignColor(clone, new_color);
      external_replacements[instruction] = clone;
      clone->set_sharding(instruction->sharding_ptr());
    };

    int64_t index = 0;
    for (auto* instruction : config.external_roots) {
      clone_root(instruction, index++);
    }

    for (auto* instruction : config.internal_roots) {
      clone_root(instruction, index++);
    }
  }

  auto* task_root = task->called_computations()[0]->root_instruction();
  for (auto* user : task->users()) {
    auto* operand = task_root->operand(user->tuple_index());
    auto* clone = external_replacements.at(operand);
    clone->set_sharding(user->sharding_ptr());
    TF_RETURN_IF_ERROR(parent->ReplaceInstruction(user, clone));
  }

  TF_RETURN_IF_ERROR(parent->parent()->RemoveEmbeddedComputation(computation));

  return new_colors;
}

}  // namespace xla
