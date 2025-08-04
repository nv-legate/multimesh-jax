/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_loop_unroll.h"

#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "loop_scheduler.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/loop_scheduler.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_loop.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/tsl/platform/default/statusor.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace {

// Returns whether an `instruction` is on the critical
// path in a loop. The definition if critical is not exact
// and instead uses heuristics to estimate such as
// whether any other task uses the outputs or if
// the outputs go directly to the root tuple.
bool IsNonCritical(HloInstruction* instruction) {
  for (auto* user : instruction->users()) {
    if (user->users().size() != 1) {
      VLOG(5) << instruction->name() << ":" << instruction->to_apply()->name()
              << " is critical through multiple users";
      for (auto* u : user->users()) {
        VLOG(5) << "    " << user->name() << " used by " << u->name();
      }
      return false;
    }
    for (auto* user : user->users()) {
      if (user != instruction->parent()->root_instruction()) {
        VLOG(5) << instruction->name() << ":" << instruction->to_apply()->name()
                << " is critical through non-root user " << user->name();
        return false;
      }
    }
  }
  return true;
}

}  // namespace

struct LoopInstructionKey {
  const HloInstruction* instruction;
  int64_t iteration;
};

bool operator==(const LoopInstructionKey& lhs, const LoopInstructionKey& rhs) {
  return lhs.instruction == rhs.instruction && lhs.iteration == rhs.iteration;
}

template <typename H>
H AbslHashValue(H h, const LoopInstructionKey& key) {
  return H::combine(std::move(h), (intptr_t)key.instruction, key.iteration);
}

struct LoopCarriedKey {
  const HloInstruction* call;
  int64_t tuple_index;
};

bool operator==(const LoopCarriedKey& lhs, const LoopCarriedKey& rhs) {
  return lhs.call == rhs.call && lhs.tuple_index == rhs.tuple_index;
}

template <typename H>
H AbslHashValue(H h, const LoopCarriedKey& key) {
  return H::combine(std::move(h), (intptr_t)key.call, key.tuple_index);
}

bool MpmdLoopUnroll::IsLoopDependent(const HloInstruction* instr) {
  auto color = Color(instr);
  if (!color.has_value()) {
    return false;
  }

  auto config = partition_->ConfigForColor(*color);
  return config.task_options.loop_dependent_devices != nullptr;
}

absl::Status MpmdLoopUnroll::ApplyLoopDependentColor(
    HloInstruction* instr, int iter, ClonedColorMap& cloned_colors) {
  // get the loop dependent devices callback
  auto color_opt = Color(instr);
  if (!color_opt.has_value()) {
    return InvalidArgumentStrCat("instruction ", instr->name(),
                                 " expected to have color");
  }

  auto color = *color_opt;
  HloPartition::ColorConfig config = partition_->ConfigForColor(color);
  multimesh::TaskOptions task_options = config.task_options;
  auto device_callback = task_options.loop_dependent_devices;
  if (device_callback == nullptr) {
    return InvalidArgumentStrCat(
        "color ", color, " expected to have non-null loop_dependent_devices");
  }

  // get devices/task options for iteration
  std::vector<int64_t> device_list = device_callback(iter);
  const int64_t start = device_list.front();
  const int64_t num_devices = device_list.size();
  zuku::DeviceList zuku_devices{{.start = start, .num_devices = num_devices}};

  // ensure iteration device count same as original color's device count
  if (num_devices != config.devices.size()) {
    LOG(FATAL) << "loop dependent device count for color " << color << " has "
               << num_devices << " devices for iter " << iter << " different "
               << "from original " << config.devices.size() << " devices";
  }

  // get and assign color for iteration (use existing color if already exists)
  std::string iter_color;
  if (cloned_colors.contains(color) &&
      cloned_colors[color].contains(zuku_devices)) {
    iter_color = cloned_colors[color][zuku_devices];
  } else {
    // create new color that is no longer loop dependent
    std::string iter_color_name =
        absl::StrCat(color, "_submesh_[", start, ",", start + num_devices, ")");
    task_options.loop_dependent_devices = nullptr;
    TF_ASSIGN_OR_RETURN(
        iter_color, partition_->CloneColor(color, iter_color_name, zuku_devices,
                                           task_options));

    cloned_colors[color][zuku_devices] = iter_color;
  }

  AssignColor(instr, iter_color);

  VLOG(3) << "Assigned loop-dependent color " << iter_color << " to "
          << instr->name();

  return absl::OkStatus();
}

absl::Status MpmdLoopUnroll::Unroll(HloInstruction* while_loop,
                                    ClonedColorMap& cloned_colors,
                                    const InstructionProperties& properties,
                                    HloPassCleanup& cleanup) {
  HloComputation* parent = while_loop->parent();
  HloComputation* body = while_loop->called_computations()[0];
  HloInstruction* input_tuple = while_loop->mutable_operand(0);

  TF_ASSIGN_OR_RETURN(
      auto config,
      GetMicrobatchConfig(std::string(while_loop->name()),
                          while_loop->raw_backend_config_string()));

  absl::flat_hash_map<LoopInstructionKey, HloInstruction*> clone_map;

  auto get_clone = [&](const HloInstruction* instruction,
                       int64_t iteration) -> absl::StatusOr<HloInstruction*> {
    auto iter = clone_map.find({instruction, iteration});
    if (iter == clone_map.end()) {
      return InvalidArgumentStrCat(instruction->name(), " for iteration ",
                                   iteration, " has no clone");
    }
    return iter->second;
  };

  // lambda to determine if an instruction can have a scheduling name
  absl::flat_hash_set<HloInstruction*> was_loop_dependent;
  auto never_loop_dependent = [&](const HloInstruction* instruction) -> bool {
    // unrolled instructions lose loop dependency but still not schedulable
    return !was_loop_dependent.contains(instruction) &&
           !IsLoopDependent(instruction);
  };

  std::vector<std::vector<HloInstruction*>> tasks(config.num_iterations);

  HloInstruction* parameter_tuple = body->parameter_instruction(0);

  HloInstruction* prev_loop_output_tuple = input_tuple;
  std::vector<HloInstruction*> dummy_tuples;

  for (int iter = 0; iter < config.num_iterations; ++iter) {
    int64_t stage = 0;
    for (auto* instruction : body->MakeInstructionPostOrder()) {
      // track if clone is an input_tuple operand, meaning it's outside of the
      // while loop and should not have a loop dependent color applied to it
      bool clone_is_input_operand = false;
      switch (instruction->opcode()) {
        case HloOpcode::kParameter:
          break;
        case HloOpcode::kGetTupleElement: {
          const auto& input_properties =
              properties.Get(input_tuple->operand(instruction->tuple_index()));
          if (instruction->operand(0) == parameter_tuple) {
            if (input_properties.loop_carried_index.has_value()) {
              HloInstruction* alias = prev_loop_output_tuple->mutable_operand(
                  instruction->tuple_index());
              if (never_loop_dependent(alias)) {
                alias->set_metadata_scheduling_name(
                    input_tuple->operand(instruction->tuple_index())->name());
              }
              clone_map[{instruction, iter}] = alias;
              clone_is_input_operand = prev_loop_output_tuple == input_tuple;
            } else {
              HloInstruction* alias =
                  input_tuple->mutable_operand(instruction->tuple_index());
              clone_map[{instruction, iter}] = alias;
              clone_is_input_operand = true;
            }
          } else {
            VLOG(5) << "cloning tuple element " << instruction->name()
                    << " of tuple " << instruction->operand(0)->name();
            TF_ASSIGN_OR_RETURN(auto* new_operand,
                                get_clone(instruction->operand(0), iter));
            HloInstruction* clone =
                parent->AddInstruction(HloInstruction::CreateGetTupleElement(
                    new_operand, instruction->tuple_index()));
            clone_map[{instruction, iter}] = clone;
            if (instruction->has_sharding()) {
              clone->set_sharding(instruction->sharding_ptr());
            }
            PropagateColor(instruction, clone);
          }
          break;
        }
        case HloOpcode::kCall: {
          if (partition_->IsNonCritical(instruction) &&
              !IsNonCritical(instruction)) {
            return InvalidArgumentStrCat(
                instruction->name(), "->", instruction->to_apply()->name(),
                " tagged as non-critical, but does not satisfy condition");
          }
          std::vector<HloInstruction*> new_operands;
          new_operands.reserve(instruction->operand_count());
          for (auto* operand : instruction->operands()) {
            VLOG(5) << "cloning operand " << operand->name() << " of call "
                    << instruction->name();
            TF_ASSIGN_OR_RETURN(auto* new_operand, get_clone(operand, iter));
            new_operands.push_back(new_operand);
          }
          HloInstruction* clone =
              parent->AddInstruction(HloInstruction::CreateCall(
                  instruction->shape(), new_operands, instruction->to_apply()));
          PropagateColor(instruction, clone);
          clone_map[{instruction, iter}] = clone;
          tasks[iter].push_back(clone);
          ++stage;
          break;
        }
        case HloOpcode::kTuple: {
          if (instruction != body->root_instruction()) {
            return InvalidArgumentStrCat(
                "loop ", while_loop->name(), " has tuple instruction ",
                instruction->name(), " that is not a root tuple");
          }
          std::vector<HloInstruction*> new_operands;
          new_operands.reserve(instruction->operand_count());
          for (int64_t index = 0; index < instruction->operand_count();
               ++index) {
            auto* operand = instruction->operand(index);
            VLOG(5) << "cloning operand " << operand->name() << " of tuple "
                    << instruction->name();
            TF_ASSIGN_OR_RETURN(auto* new_operand, get_clone(operand, iter));
            const auto& operand_properties = properties.Get(operand);
            if (operand_properties.loop_carried_index.has_value() &&
                never_loop_dependent(new_operand)) {
              new_operand->set_metadata_scheduling_name(
                  input_tuple->operand(index)->name());
            }
            new_operands.push_back(new_operand);
          }
          HloInstruction* clone =
              parent->AddInstruction(HloInstruction::CreateTuple(new_operands));
          clone->set_sharding(instruction->sharding_ptr());
          prev_loop_output_tuple = clone;
          dummy_tuples.push_back(clone);
          break;
        }
        case HloOpcode::kCustomCall: {
          if (instruction->IsCustomCall(kCustomCallSliceOffset)) {
            if (!clone_map.contains({instruction, iter})) {
              HloInstruction* clone =
                  parent->AddInstruction(instruction->Clone());
              int offset = iter * config.microbatch_size;
              AddAttribute(clone, "offset", offset);
              clone_map[{instruction, iter}] = clone;
            }
            break;
          } else if (instruction->IsCustomCall(kCustomCallDummyOperation)) {
            // pass
            HloInstruction* clone =
                parent->AddInstruction(instruction->Clone());
            clone_map[{instruction, iter}] = clone;
            break;
          }
        }
        default:
          return InvalidArgumentStrCat(
              "loop ", while_loop->name(), " has unsupported instruction ",
              instruction->name(), " of opcode ",
              HloOpcodeString(instruction->opcode()), " when unrolling loops");
      }

      // modify color of newly created instruction if its mesh is loop
      // dependent and within loop body
      if (clone_map.contains({instruction, iter})) {
        HloInstruction* clone = clone_map[{instruction, iter}];
        if (IsLoopDependent(clone) && !clone_is_input_operand) {
          was_loop_dependent.insert(clone);
          TF_RETURN_IF_ERROR(
              ApplyLoopDependentColor(clone, iter, cloned_colors));
        }
      }
    }
  }

  for (auto* user : while_loop->users()) {
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(
        prev_loop_output_tuple->mutable_operand(user->tuple_index())));
    cleanup.RemoveInstruction(user);
  }

  for (auto* tuple : dummy_tuples) {
    cleanup.RemoveInstruction(tuple);
  }

  TF_ASSIGN_OR_RETURN(auto schedule, ScheduleLoops(*partition_, config, tasks));

  // add control dependencies across the schedule
  for (int64_t stage = 0; stage < schedule.size(); ++stage) {
    for (int64_t lhs = 0; lhs < schedule[stage].size(); ++lhs) {
      // these dependencies are only meaningful if the device lists overlap
      zuku::DeviceList lhs_devices =
          partition_->DevicesForInstruction(schedule[stage][lhs]);
      for (int64_t rhs = lhs + 1; rhs < schedule[stage].size(); ++rhs) {
        zuku::DeviceList rhs_devices =
            partition_->DevicesForInstruction(schedule[stage][rhs]);
        if (lhs_devices == rhs_devices) {
          TF_RETURN_IF_ERROR(schedule[stage][lhs]->AddControlDependencyTo(
              schedule[stage][rhs]));
          break;
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(while_loop->ReplaceAllUsesWith(prev_loop_output_tuple));
  cleanup.RemoveInstruction(while_loop);
  cleanup.RemoveInstruction(input_tuple);
  return absl::OkStatus();
}

absl::StatusOr<bool> MpmdLoopUnroll::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto properties = InstructionProperties::Create(module);
  HloPassCleanup cleanup;
  bool changed = false;

  ClonedColorMap cloned_colors;

  HloComputation* entry_computation = module->entry_computation();
  for (auto* instruction : entry_computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kWhile &&
        instruction->has_backend_config()) {
      changed = true;
      TF_RETURN_IF_ERROR(
          Unroll(instruction, cloned_colors, properties, cleanup));

      // remove body and condition computations associated with loop
      cleanup.RemoveComputation(instruction->called_computations()[0]);
      cleanup.RemoveComputation(instruction->called_computations()[1]);
    }
  }

  TF_RETURN_IF_ERROR(cleanup.CleanUp());
  return changed;
}

}  // namespace xla
