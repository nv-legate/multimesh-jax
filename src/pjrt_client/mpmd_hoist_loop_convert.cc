/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_hoist_loop_convert.h"

#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"

namespace xla {

absl::StatusOr<HloComputation*> MpmdHoistLoopConvert::CloneArgTupleComputation(
    HloComputation* computation, HloModule* module,
    const std::vector<Shape>& arg_tuple_shapes) {
  HloComputation::Builder builder{computation->name()};
  HloCloneContext context{module};

  auto* old_arg_tuple = computation->parameter_instruction(0);
  TF_ASSIGN_OR_RETURN(
      auto* new_arg_tuple,
      builder.AddParameter(HloInstruction::CreateParameter(
          0, Shape{PrimitiveType::TUPLE, {}, {}, arg_tuple_shapes},
          old_arg_tuple->name())));
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> clone_map;
  clone_map[old_arg_tuple] = new_arg_tuple;

  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction == old_arg_tuple) {
      continue;
    } else if (instruction->opcode() == HloOpcode::kGetTupleElement &&
               instruction->operand(0) == old_arg_tuple) {
      auto* new_gte =
          builder.AddInstruction(HloInstruction::CreateGetTupleElement(
              new_arg_tuple, instruction->tuple_index()));
      PropagateProperties(instruction, new_gte);
      clone_map[instruction] = new_gte;
    } else if (instruction->opcode() == HloOpcode::kConvert) {
      // this is now a no-op convert, skip it
      auto iter = clone_map.find(instruction->operand(0));
      if (iter == clone_map.end()) {
        return InvalidArgumentStrCat(instruction->operand(0)->name(),
                                     " not in clone map for while loop ",
                                     instruction->name());
      }

      if (ShapeUtil::Compatible(iter->second->shape(), instruction->shape())) {
        // don't clone this, just point user instructions to use the
        // operand directly
        clone_map[instruction] = iter->second;
      } else {
        clone_map[instruction] =
            builder.AddInstruction(instruction->CloneWithNewOperands(
                instruction->shape(), {iter->second}));
      }
    } else if (instruction->opcode() == HloOpcode::kTuple &&
               instruction->users().size() == 1 &&
               instruction->users().front()->opcode() ==
                   HloOpcode::kOptimizationBarrier) {
      std::vector<HloInstruction*> new_operands;
      for (auto* operand : instruction->operands()) {
        auto iter = clone_map.find(operand);
        if (iter == clone_map.end()) {
          return InvalidArgumentStrCat(operand->name(),
                                       " not in clone map for computation ",
                                       computation->name());
        }
        new_operands.push_back(iter->second);
      }
      auto* new_tuple =
          builder.AddInstruction(HloInstruction::CreateTuple(new_operands));
      clone_map[instruction] = new_tuple;
    } else if (instruction->opcode() == HloOpcode::kOptimizationBarrier &&
               instruction->shape().IsTuple()) {
      auto iter = clone_map.find(instruction->operand(0));
      if (iter == clone_map.end()) {
        return InvalidArgumentStrCat(instruction->operand(0)->name(),
                                     " not in clone map for computation ",
                                     computation->name());
      }
      auto* new_barrier = builder.AddInstruction(HloInstruction::CreateUnary(
          iter->second->shape(), HloOpcode::kOptimizationBarrier,
          iter->second));
      clone_map[instruction] = new_barrier;
    } else {
      absl::InlinedVector<HloInstruction*, 2> new_operands;
      for (auto* operand : instruction->operands()) {
        auto iter = clone_map.find(operand);
        if (iter == clone_map.end()) {
          return InvalidArgumentStrCat(operand->name(),
                                       " not in clone map for computation ",
                                       computation->name());
        }
        new_operands.push_back(iter->second);
      }

      auto shape = [&] {
        if (instruction->opcode() == HloOpcode::kGetTupleElement) {
          return new_operands[0]->shape().tuple_shapes(
              instruction->tuple_index());
        }
        return instruction->shape();
      }();

      auto* clone = builder.AddInstruction(
          instruction->CloneWithNewOperands(std::move(shape), new_operands));
      clone_map[instruction] = clone;
    }
  }

  return module->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                   /*is_entry=*/false);
}

// Search through all users and potential opt-barrier alises to find
// all uses of the instruction
absl::InlinedVector<HloInstruction*, 6> FindAllUsers(
    HloInstruction* instruction) {
  absl::InlinedVector<HloInstruction*, 4> to_visit = {instruction};
  absl::InlinedVector<HloInstruction*, 6> users;
  while (!to_visit.empty()) {
    HloInstruction* next = to_visit.back();
    to_visit.pop_back();

    for (auto* user : next->users()) {
      if (user->opcode() == HloOpcode::kTuple && user->users().size() == 1 &&
          user->users().front()->opcode() == HloOpcode::kOptimizationBarrier) {
        auto* barrier = user->users().front();
        const int64_t operand_index = user->operand_index(next);
        for (auto* gte : barrier->users()) {
          if (gte->tuple_index() == operand_index) {
            to_visit.push_back(gte);
          }
        }
      } else {
        users.push_back(user);
      }
    }
  }
  return users;
}

// Returns the root index if the users are the root and only equivalent converts
bool UsersAreRootAndEqivalentConverts(HloInstruction* instruction,
                                      HloInstruction* root) {
  auto all_users = FindAllUsers(instruction);
  if (all_users.size() < 2) {
    VLOG(5) << instruction->name() << ":" << instruction->shape()
            << " does not have enough users";
    return false;
  }

  bool found_direct_convert_user = false;
  for (auto* user : instruction->users()) {
    if (user->opcode() == HloOpcode::kConvert) {
      found_direct_convert_user = true;
      break;
    }
  }

  if (!found_direct_convert_user) {
    VLOG(5) << instruction->name() << ":" << instruction->shape()
            << " does not have a direct convert user";
    return false;
  }

  bool root_found = false;
  std::optional<PrimitiveType> convert_type{std::nullopt};

  for (auto* user : all_users) {
    if (user == root) {
      root_found = true;
    } else if (user->opcode() != HloOpcode::kConvert) {
      VLOG(5) << instruction->name() << ":" << instruction->shape()
              << " has a user " << user->name() << " that is not a convert";
      return false;
    } else {
      auto allowed_type = convert_type.value_or(user->shape().element_type());
      if (allowed_type != user->shape().element_type()) {
        VLOG(5) << instruction->name() << " has converts with different types";
        return false;
      }
      convert_type = user->shape().element_type();
    }
  }
  if (!root_found) {
    VLOG(5) << instruction->name() << ":" << instruction->shape()
            << " is not a repeated loop parameter";
  }

  return root_found;
}

absl::StatusOr<bool> MpmdHoistLoopConvert::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloComputation*> to_visit = {module->entry_computation()};

  struct HoistableConvert {
    HloInstruction* tuple_input;
    HloInstruction* loop_gte;
    HloInstruction* convert;
  };

  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      absl::flat_hash_map<int64_t, HoistableConvert> hoistable_converts;
      if (instruction->opcode() == HloOpcode::kWhile) {
        auto* body = instruction->called_computations()[0];
        auto* body_root = body->root_instruction();
        auto* condition = instruction->called_computations()[1];
        auto* body_arg_tuple = body->parameter_instruction(0);
        auto* condition_arg_tuple = condition->parameter_instruction(0);
        auto* input_tuple = instruction->mutable_operand(0);
        for (auto* user : body_arg_tuple->users()) {
          auto* loop_input = input_tuple->mutable_operand(user->tuple_index());
          if (loop_input->opcode() == HloOpcode::kParameter &&
              UsersAreRootAndEqivalentConverts(user, body_root)) {
            HloInstruction* convert = [=] {
              for (auto* i : user->users()) {
                if (i->opcode() == HloOpcode::kConvert) {
                  return i;
                }
              }
              return (HloInstruction*)nullptr;
            }();

            CHECK(convert != nullptr);

            VLOG(5) << "found hoistable input " << user->name() << " for loop "
                    << instruction->name();
            hoistable_converts[user->tuple_index()] = {
                .tuple_input =
                    input_tuple->mutable_operand(user->tuple_index()),
                .loop_gte = user,
                .convert = convert};
          } else {
            VLOG(5) << "loop input " << loop_input->name() << ":"
                    << user->name() << " " << loop_input->shape()
                    << " at index " << user->tuple_index()
                    << " is not a parameter";
          }
        }

        bool condition_can_be_hoisted = true;
        for (auto* user : condition_arg_tuple->users()) {
          // converts in the body must also be hoistable in the condition!
          if (hoistable_converts.contains(user->tuple_index())) {
            auto* loop_input =
                input_tuple->mutable_operand(user->tuple_index());
            if (user->users().size() > 1 ||
                (user->users().size() == 1 &&
                 user->users().front()->opcode() != HloOpcode::kConvert)) {
              VLOG(5) << "condition for " << instruction->name()
                      << " cannot be hoisted";
              condition_can_be_hoisted = false;
              break;
            }
          }
        }

        if (!condition_can_be_hoisted) {
          // go to the next instruction
          continue;
        }

        for (auto* user : instruction->users()) {
          // any output of the while loop cannot be hoised since the unconverted
          // output is used downstream
          if (user != computation->root_instruction() &&
              user->users().empty()) {
            // oh, this is never actually used
            TF_RETURN_IF_ERROR(computation->RemoveInstruction(user));
          } else if (hoistable_converts.contains(user->tuple_index())) {
            VLOG(5) << "output " << user->name()
                    << " prevents hoisting convert at index "
                    << user->tuple_index();
            hoistable_converts.erase(user->tuple_index());
          }
        }

        // all of the converts are used downstream unconverted
        if (hoistable_converts.empty()) {
          continue;
        }

        changed = true;

        std::vector<Shape> new_tuple_shapes;
        std::vector<HloInstruction*> new_input_tuple_operands;

        new_input_tuple_operands.reserve(input_tuple->operand_count());

        for (int64_t index = 0; index < input_tuple->operand_count(); ++index) {
          if (hoistable_converts.contains(index)) {
            const auto& hoisted_convert = hoistable_converts[index];
            VLOG(5) << "hoisting loop input " << index << " with conversion "
                    << hoisted_convert.tuple_input->shape() << "->"
                    << hoisted_convert.convert->shape();

            if (zero_out_arguments_) {
              auto* constant = computation->AddInstruction(
                  HloInstruction::CreateConstant(LiteralUtil::CreateR0(
                      hoisted_convert.convert->shape().element_type(), 0.0)));
              auto* bcast_to_shape =
                  computation->AddInstruction(HloInstruction::CreateBroadcast(
                      hoisted_convert.convert->shape(), constant, {}));
              auto* copy =
                  computation->AddInstruction(HloInstruction::CreateUnary(
                      bcast_to_shape->shape(), HloOpcode::kCopy,
                      bcast_to_shape));
              PropagateProperties(hoisted_convert.tuple_input, copy);
              new_input_tuple_operands.push_back(copy);
            } else {
              auto* new_convert =
                  computation->AddInstruction(HloInstruction::CreateConvert(
                      hoisted_convert.convert->shape(),
                      hoisted_convert.tuple_input));
              // force a copy of the convert to make it clear this should not be
              // recomputed anywhere
              auto* copy =
                  computation->AddInstruction(HloInstruction::CreateUnary(
                      new_convert->shape(), HloOpcode::kCopy, new_convert));

              PropagateProperties(hoisted_convert.tuple_input, new_convert);
              PropagateProperties(hoisted_convert.tuple_input, copy);

              new_input_tuple_operands.push_back(copy);
            }
            new_tuple_shapes.push_back(hoisted_convert.convert->shape());
          } else {
            new_input_tuple_operands.push_back(
                input_tuple->mutable_operand(index));
            new_tuple_shapes.push_back(input_tuple->operand(index)->shape());
          }
        }

        TF_ASSIGN_OR_RETURN(
            auto* new_body,
            CloneArgTupleComputation(body, module, new_tuple_shapes));
        TF_ASSIGN_OR_RETURN(
            auto* new_condition,
            CloneArgTupleComputation(condition, module, new_tuple_shapes));
        auto* new_tuple = computation->AddInstruction(
            HloInstruction::CreateTuple(new_input_tuple_operands));
        auto* new_while =
            computation->AddInstruction(HloInstruction::CreateWhile(
                new_tuple->shape(), new_condition, new_body, new_tuple));
        new_while->set_raw_backend_config_string(
            instruction->raw_backend_config_string());

        TF_RETURN_IF_ERROR(
            instruction->ReplaceAllUsesWithDifferentShape(new_while));

        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(input_tuple));
        TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(body));
        TF_RETURN_IF_ERROR(module->RemoveEmbeddedComputation(condition));
      }
    }
  }
  return changed;
}

}  // namespace xla
