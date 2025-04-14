/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_instruction_delay_recolor.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

namespace {

bool IsParameterOrParameterCopy(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kParameter ||
         (instruction->opcode() == HloOpcode::kCopy &&
          instruction->operand(0)->opcode() == HloOpcode::kParameter);
}

// Returns any user of `instruction` that is a call or
// nullptr if no users are call instructions.
HloInstruction* GetAnyUserCall(HloInstruction* instruction) {
  for (auto* user : instruction->users()) {
    if (user->opcode() == HloOpcode::kCall) {
      return user;
    }
  }
  return nullptr;
}

}  // namespace

absl::StatusOr<std::vector<HloInstruction*>>
MpmdInstructionDelayRecolor::MaybeRecolorOutput(
    HloInstruction* output, HloInstruction* call, HloComputation* parent,
    const absl::flat_hash_map<HloInstruction*, int64_t>& depth,
    const InstructionProperties& properties) {
  std::vector<HloInstruction*> user_tree;

  if (output->users().size() != 1) {
    return user_tree;
  }

  HloInstruction* user = output->users().front();
  if (user->opcode() == HloOpcode::kTuple) {
    return user_tree;
  }

  auto output_color = Color(call);
  auto user_color = Color(user);

  if (!output_color.has_value() || !user_color.has_value()) {
    return user_tree;
  }

  if (!depth.contains(user)) {
    LOG(FATAL) << user->name() << " not in map";
  }

  if (!depth.contains(call)) {
    LOG(FATAL) << call->name() << " not in map";
  }

  const int64_t reuse_distance = depth.at(user) - depth.at(call);
  if (reuse_distance < 2) {
    // don't bother, this is the responsibility of the fusion pass
    return user_tree;
  }

  auto output_devices = partition_->DevicesForColor(*output_color);
  auto user_devices = partition_->DevicesForColor(*user_color);

  if (output_devices != user_devices) {
    VLOG(5) << output->name()
            << " has mismatched devices for user: " << output_devices
            << " != " << user_devices;
    return user_tree;
  }

  auto* matching_root =
      call->called_computations()[0]->root_instruction()->mutable_operand(
          output->tuple_index());

  std::vector<HloInstruction*> to_visit = {matching_root};

  while (!to_visit.empty()) {
    HloInstruction* next = to_visit.back();
    to_visit.pop_back();

    if (next->users().size() != 1) {
      // okay to add derived constants and parameters, we will copy these
      if (!properties.DerivedInput(next)) {
        VLOG(5) << next->name() << " breaks tree for moving " << output->name();
        to_visit.clear();
        user_tree.clear();
        break;
      }
    }

    VLOG(5) << next->name() << " migrating in user tree from " << call->name()
            << " to " << user->name();
    user_tree.push_back(next);

    for (auto* operand : next->mutable_operands()) {
      to_visit.push_back(operand);
    }
  }

  return user_tree;
}

absl::StatusOr<bool> MpmdInstructionDelayRecolor::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // the next phase is more complicated
  // rather than looking at single instructions, see if there are "instruction
  // trees" which can be moved later to co-locate with users
  std::vector<HloComputation*> to_visit = {module->entry_computation()};

  auto properties = InstructionProperties::Create(module);
  HloPassCleanup cleanup;

  bool changed = false;
  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    absl::flat_hash_map<HloInstruction*, int64_t> depth;

    auto postorder = computation->MakeInstructionPostOrder();
    for (auto* instruction : postorder) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        to_visit.push_back(instruction->called_computations()[0]);
        depth[instruction] = depth.size();
      } else if (instruction->opcode() != HloOpcode::kGetTupleElement) {
        depth[instruction] = depth.size();
      }
    }

    absl::flat_hash_map<
        HloInstruction*,
        std::vector<std::pair<HloInstruction*, HloInstruction*>>>
        to_fuse;
    HloCloneContext context{module};
    std::vector<HloInstruction*> calls_to_remove;
    absl::flat_hash_set<HloInstruction*> outputs_removed;
    for (auto* instruction : postorder) {
      if (instruction->opcode() == HloOpcode::kCall) {
        HloComputation* called_comp = instruction->called_computations()[0];
        auto color = Color(instruction);
        if (!color.has_value()) {
          return InvalidArgumentStrCat(instruction->name(),
                                       " was not assigned a color");
        }

        absl::flat_hash_set<HloInstruction*> to_remove;

        std::vector<HloInstruction*> users_to_merge;
        absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
            outputs_to_visit;

        for (auto* output : instruction->users()) {
          auto* user_call = GetAnyUserCall(output);
          // there are no calls that use this output
          if (user_call == nullptr) {
            continue;
          }

          TF_ASSIGN_OR_RETURN(
              std::vector<HloInstruction*> to_move,
              MaybeRecolorOutput(output, instruction, computation, depth,
                                 properties));

          if (!to_move.empty()) {
            VLOG(5) << "fusing " << to_move.size() << " instructions into "
                    << user_call->name() << ", removing from "
                    << instruction->name();
            // the tree is latest first and should be reversed
            for (auto iter = to_move.rbegin(); iter != to_move.rend(); ++iter) {
              to_fuse[user_call].emplace_back(*iter, instruction);
            }
            to_remove.insert(to_move.begin(), to_move.end());
            outputs_removed.insert(output);
            changed = true;
          }
        }

        if (!to_remove.empty() || to_fuse.contains(instruction)) {
          calls_to_remove.push_back(instruction);
          auto* called_comp = instruction->called_computations()[0];
          auto* call_root_tuple = called_comp->root_instruction();
          VLOG(5) << "modifying " << instruction->name() << " with computation "
                  << called_comp->name();
          to_remove.insert(call_root_tuple);
          absl::flat_hash_map<HloInstruction*, HloInstruction*> cloned_operands;
          auto& fuse_in = to_fuse[instruction];
          HloComputation::Builder builder{
              instruction->called_computations()[0]->name()};

          int64_t clone_parameter_number = 0;
          std::vector<HloInstruction*> new_call_operands;
          auto add_clone = [&](HloInstruction* to_clone,
                               HloInstruction* parent_call) -> absl::Status {
            // already added from somone else's tree
            auto iter = cloned_operands.find(to_clone);
            if (iter != cloned_operands.end()) {
              VLOG(5) << to_clone->name() << " already added to "
                      << instruction->name();
              return absl::OkStatus();
            }

            VLOG(5) << "cloning " << to_clone->name()
                    << " to new computation for " << instruction->name()
                    << " from prev parent " << parent_call->name();
            absl::InlinedVector<HloInstruction*, 2> new_operands;
            for (auto* operand : to_clone->operands()) {
              auto iter = cloned_operands.find(operand);
              if (iter == cloned_operands.end()) {
                return InvalidArgumentStrCat(operand->name(),
                                             " was not added to the clone map");
              }
              new_operands.push_back(iter->second);
            }

            if (to_clone->opcode() == HloOpcode::kParameter) {
              TF_ASSIGN_OR_RETURN(
                  auto* param,
                  builder.AddParameter(HloInstruction::CreateParameter(
                      clone_parameter_number++, to_clone->shape(),
                      to_clone->name())));
              param->set_sharding(to_clone->sharding_ptr());
              cloned_operands[to_clone] = param;
              new_call_operands.push_back(
                  parent_call->mutable_operand(to_clone->parameter_number()));
            } else {
              auto* clone =
                  builder.AddInstruction(to_clone->CloneWithNewOperands(
                      to_clone->shape(), new_operands, &context));
              clone->set_sharding(to_clone->sharding_ptr());
              cloned_operands[to_clone] = clone;
            }
            return absl::OkStatus();
          };

          for (auto [to_fuse, prev_owner] : fuse_in) {
            TF_RETURN_IF_ERROR(add_clone(to_fuse, prev_owner));
          }

          // some of the parameters may no longer be required
          for (int64_t param_number = 0;
               param_number < called_comp->num_parameters(); ++param_number) {
            auto* call_operand = instruction->mutable_operand(param_number);

            if (outputs_removed.contains(call_operand)) {
              auto* producer_root =
                  call_operand->mutable_operand(0)
                      ->called_computations()[0]
                      ->root_instruction()
                      ->mutable_operand(call_operand->tuple_index());
              auto* call_param =
                  called_comp->parameter_instruction(param_number);
              to_remove.insert(call_param);
              auto* new_operand_clone = cloned_operands[producer_root];
              VLOG(5) << producer_root->name() << " replacing as operand for "
                      << call_param->name();
              cloned_operands[call_param] = new_operand_clone;
            }
          }

          for (auto* to_clone : instruction->called_computations()[0]
                                    ->MakeInstructionPostOrder()) {
            if (properties.DerivedInput(to_clone) ||
                !to_remove.contains(to_clone)) {
              TF_RETURN_IF_ERROR(add_clone(to_clone, instruction));
            }
          }

          std::vector<HloInstruction*> new_root_operands;
          for (auto* user : instruction->users()) {
            auto* matching_root =
                call_root_tuple->mutable_operand(user->tuple_index());
            if (!outputs_removed.contains(user)) {
              auto iter = cloned_operands.find(matching_root);
              if (iter == cloned_operands.end()) {
                return InvalidArgumentStrCat(user->name(), " with alias root ",
                                             matching_root->name(),
                                             " has no clone in the map");
              }
              VLOG(5) << user->name() << " with matching root "
                      << matching_root->name() << " is still an output "
                      << instruction->name();
              new_root_operands.push_back(cloned_operands[matching_root]);
            } else {
              VLOG(5) << user->name() << " with matching root "
                      << matching_root->name() << " is no longer an output of "
                      << instruction->name();
            }
          }

          auto* new_root = builder.AddInstruction(
              HloInstruction::CreateTuple(new_root_operands));

          for (auto& [old, clone] : cloned_operands) {
            AssignColor(clone, *color);
          }

          auto* new_computation = module->AddComputationAndUnifyNamesAndIds(
              builder.Build(new_root), /*is_entry=*/false);

          auto* new_call =
              computation->AddInstruction(HloInstruction::CreateCall(
                  new_root->shape(), new_call_operands, new_computation));
          AssignColor(new_call, *std::move(color));
          int64_t gte_index = 0;
          for (auto* user : instruction->users()) {
            if (!outputs_removed.contains(user)) {
              auto* new_gte = computation->AddInstruction(
                  HloInstruction::CreateGetTupleElement(new_call, gte_index++));
              VLOG(5) << "replacing " << instruction->name() << " output "
                      << user->name() << " with " << new_gte->name();
              new_gte->set_sharding(user->sharding_ptr());
              TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(new_gte));
            }
          }
        }
      }
    }

    // remove the calls in reverse order to make sure get-tuple-elements are not
    // live
    for (auto iter = calls_to_remove.rbegin(); iter != calls_to_remove.rend();
         ++iter) {
      HloInstruction* call = *iter;
      for (auto* user : call->users()) {
        VLOG(5) << "removing output " << user->name() << " from call "
                << call->name();
        for (auto* user_user : user->users()) {
          VLOG(5) << user->name() << " still is live with user "
                  << user_user->name() << "!";
        }
        cleanup.RemoveInstruction(user);
      }
      VLOG(5) << "removing call " << call->name();
      cleanup.RemoveInstruction(call);
    }
  }

  TF_RETURN_IF_ERROR(cleanup.CleanUp());

  if (changed) {
    HloDCE dce{};
    TF_ASSIGN_OR_RETURN(bool _, dce.Run(module));
  }

  return changed;
}

}  // namespace xla
