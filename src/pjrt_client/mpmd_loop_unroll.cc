/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_loop_unroll.h"

#include <algorithm>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/loop_scheduler.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_loop.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

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

absl::Status MpmdLoopUnroll::Unroll(HloComputation* parent,
                                    HloComputation* body,
                                    HloInstruction* while_loop,
                                    HloInstruction* input_tuple,
                                    const InstructionProperties& properties,
                                    HloPassCleanup& cleanup) {
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

  std::vector<std::vector<HloInstruction*>> tasks(config.num_iterations);

  HloInstruction* parameter_tuple = body->parameter_instruction(0);

  HloInstruction* prev_loop_output_tuple = input_tuple;
  std::vector<HloInstruction*> dummy_tuples;
  for (int iter = 0; iter < config.num_iterations; ++iter) {
    for (auto* instruction : body->MakeInstructionPostOrder()) {
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
              alias->set_metadata_scheduling_name(
                  input_tuple->operand(instruction->tuple_index())->name());
              clone_map[{instruction, iter}] = alias;
            } else {
              HloInstruction* alias =
                  input_tuple->mutable_operand(instruction->tuple_index());
              clone_map[{instruction, iter}] = alias;
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
            if (operand_properties.loop_carried_index.has_value()) {
              new_operand->set_metadata_scheduling_name(
                  input_tuple->operand(index)->name());
            }
            new_operands.push_back(new_operand);
          }
          HloInstruction* clone =
              parent->AddInstruction(HloInstruction::CreateTuple(new_operands));
          clone->set_sharding(instruction->sharding_ptr());
          // clone_map[{instruction,iter}] = clone;
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
  for (int order = 1; order < schedule.size(); ++order) {
    TF_RETURN_IF_ERROR(
        schedule[order - 1]->AddControlDependencyTo(schedule[order]));
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
  std::vector<HloComputation*> to_visit = {module->entry_computation()};
  bool changed = false;
  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile &&
          instruction->has_backend_config()) {
        changed = true;
        HloInstruction* input_tuple = instruction->mutable_operand(0);
        // tag all the users of the loop with the scheduling name to track
        // loop-carried variables
        for (auto* user : instruction->users()) {
          const auto& user_properties = properties.Get(user);
          if (user_properties.loop_carried_index.has_value()) {
            user->set_metadata_scheduling_name(
                input_tuple->operand(user->tuple_index())->name());
          }
        }
        HloComputation* body = instruction->called_computations()[0];
        HloComputation* condition = instruction->called_computations()[1];
        TF_RETURN_IF_ERROR(
            Unroll(computation, instruction->called_computations()[0],
                   instruction, input_tuple, properties, cleanup));
        cleanup.RemoveComputation(body);
        cleanup.RemoveComputation(condition);
      }
    }
  }

  TF_RETURN_IF_ERROR(cleanup.CleanUp());
  return changed;
}

}  // namespace xla
