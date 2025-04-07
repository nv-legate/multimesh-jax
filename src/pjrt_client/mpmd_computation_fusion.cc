/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_computation_fusion.h"

#include <algorithm>
#include <limits>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"

namespace xla {

absl::Status MpmdComputationFusion::FuseComputations(HloComputation* parent,
                                                     HloInstruction* producer,
                                                     HloInstruction* consumer) {
  absl::flat_hash_map<HloInstruction*, int64_t> operands_added;
  absl::flat_hash_map</*consumer=*/HloInstruction*,
                      /*producer=*/HloInstruction*>
      producer_consumer_aliases;
  absl::flat_hash_map<HloInstruction*, int64_t> producer_output_indices;
  absl::flat_hash_set<HloInstruction*> producers_used_by_consumer;
  std::vector<HloInstruction*> fused_call_operands;
  std::vector<HloInstruction*> fused_params;
  std::vector<HloInstruction*> fused_roots;
  std::vector<HloInstruction*> fused_call_outputs;

  std::string color = producer->frontend_attributes().map().at("color");

  HloComputation* producer_call = producer->called_computations()[0];
  HloComputation* consumer_call = consumer->called_computations()[0];
  HloInstruction* producer_root = producer_call->root_instruction();
  HloInstruction* consumer_root = consumer_call->root_instruction();

  int64_t producer_param_number = 0;
  for (auto* operand : producer->mutable_operands()) {
    fused_params.push_back(
        producer_call->parameter_instruction(producer_param_number));
    fused_call_operands.push_back(operand);
    operands_added[operand] = producer_param_number;
    ++producer_param_number;
  }

  for (auto* user : producer->users()) {
    producer_output_indices[user] = user->tuple_index();
  }

  int64_t consumer_param_number = 0;
  std::vector<HloInstruction*> producer_gte_to_delete;
  for (auto* operand : consumer->mutable_operands()) {
    if (producer_output_indices.contains(operand)) {
      if (operand->users().size() == 1) {  // the consumer is the only user,
                                           // this is no longer a task output
        VLOG(5) << operand->name() << " " << operand
                << " is no longer needed as an output";
        producer_gte_to_delete.push_back(operand);
      }
      const int64_t producer_index = producer_output_indices[operand];
      VLOG(5) << "consumer " << operand->name() << " uses producer output "
              << producer_index;
      producer_consumer_aliases[consumer_call->parameter_instruction(
          consumer_param_number)] =
          producer_root->mutable_operand(producer_index);
      producers_used_by_consumer.insert(operand);
    } else if (operands_added.contains(operand)) {
      // don't add this twice, we only need a single operand
      producer_consumer_aliases[consumer_call->parameter_instruction(
          consumer_param_number)] =
          producer_call->parameter_instruction(operands_added[operand]);
    } else {
      // not shared by anyone
      fused_params.push_back(
          consumer_call->parameter_instruction(consumer_param_number));
      fused_call_operands.push_back(operand);
    }
    ++consumer_param_number;
  }

  for (auto* user : producer->users()) {
    // if someone else besides the consumer uses the output
    if (user->users().size() > 1 ||
        !producers_used_by_consumer.contains(user)) {
      fused_roots.push_back(
          producer_root->mutable_operand(user->tuple_index()));
      fused_call_outputs.push_back(user);
      VLOG(5) << "producer still produces " << user->name() << " : "
              << fused_roots.back()->name();
    }
  }

  for (auto* user : consumer->users()) {
    fused_roots.push_back(consumer_root->mutable_operand(user->tuple_index()));
    VLOG(5) << "consumer still produces " << user->name() << " : "
            << fused_roots.back()->name();
    fused_call_outputs.push_back(user);
  }

  HloCloneContext context{parent->parent()};
  HloComputation::Builder builder{consumer_call->name()};

  int64_t fused_param_number = 0;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> clone_map;
  for (auto* param : fused_params) {
    TF_ASSIGN_OR_RETURN(
        auto* new_param,
        builder.AddParameter(HloInstruction::CreateParameter(
            fused_param_number, param->shape(), param->name())));
    new_param->set_sharding(param->sharding_ptr());
    PropagateProperties(param, new_param);
    clone_map[param] = new_param;
    ++fused_param_number;
  }

  for (auto* instruction : producer_call->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kParameter &&
        instruction != producer_root) {
      absl::InlinedVector<HloInstruction*, 2> cloned_operands;
      for (auto* operand : instruction->operands()) {
        if (!clone_map.contains(operand)) {
          return InvalidArgumentStrCat(
              operand->name(), " was never added to clone map when fusing ",
              producer->name(), " and ", consumer->name());
        }
        cloned_operands.push_back(clone_map[operand]);
      }
      auto* new_instruction =
          builder.AddInstruction(instruction->CloneWithNewOperands(
              instruction->shape(), cloned_operands, &context));
      clone_map[instruction] = new_instruction;
    }
  }

  for (auto* instruction : consumer_call->MakeInstructionPostOrder()) {
    if (instruction->opcode() != HloOpcode::kParameter &&
        instruction != consumer_root) {
      absl::InlinedVector<HloInstruction*, 2> cloned_operands;
      for (auto* operand : instruction->operands()) {
        auto* actual_operand = [&] {
          if (producer_consumer_aliases.contains(operand)) {
            VLOG(5) << "consumer operand " << operand->name()
                    << " is now internal instruction";
            return producer_consumer_aliases[operand];
          }
          return operand;
        }();
        if (!clone_map.contains(actual_operand)) {
          return InvalidArgumentStrCat(
              actual_operand->name(),
              " was never added to clone map when fusing ", producer->name(),
              " and ", consumer->name());
        }
        cloned_operands.push_back(clone_map[actual_operand]);
      }
      auto* new_instruction =
          builder.AddInstruction(instruction->CloneWithNewOperands(
              instruction->shape(), cloned_operands, &context));
      clone_map[instruction] = new_instruction;
    }
  }

  std::vector<HloInstruction*> fused_root_operands;
  for (auto* root : fused_roots) {
    auto iter = clone_map.find(root);
    if (iter == clone_map.end()) {
      return InvalidArgumentStrCat("fused root ", root->name(),
                                   " is not in clone map");
    }
    fused_root_operands.push_back(iter->second);
  }
  auto* root_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple(fused_root_operands));

  auto* new_comp = parent->parent()->AddComputationAndUnifyNamesAndIds(
      builder.Build(root_tuple), /*is_entry=*/false);

  auto* new_call = parent->AddInstruction(HloInstruction::CreateCall(
      root_tuple->shape(), fused_call_operands, new_comp));

  PropagateColor(producer, new_call);

  // these need to be remapped to new get-tuple-element ops
  int64_t fused_output_tuple_index = 0;
  for (auto* output : fused_call_outputs) {
    auto* new_gte =
        parent->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_call, fused_output_tuple_index));
    new_gte->set_sharding(output->sharding_ptr());
    PropagateProperties(output, new_gte);
    VLOG(5) << "replacing old output " << output->name() << " with new output "
            << new_gte->name();
    TF_RETURN_IF_ERROR(output->ReplaceAllUsesWith(new_gte));
    TF_RETURN_IF_ERROR(parent->RemoveInstruction(output));
    ++fused_output_tuple_index;
  }

  TF_RETURN_IF_ERROR(parent->RemoveInstruction(consumer));

  // these instructions are now internal to the fused task
  // and are no longer needed as outputs from the producer
  for (auto* instruction : producer_gte_to_delete) {
    VLOG(5) << "removing unneeded output " << instruction->name();
    TF_RETURN_IF_ERROR(parent->RemoveInstruction(instruction));
  }

  return parent->RemoveInstruction(producer);
}

bool MpmdComputationFusion::FusionMatch(const HloInstruction* lhs,
                                        const HloInstruction* rhs) {
  if (type_ == FusionType::kMatchingColor) {
    auto lhs_color = Color(lhs);
    auto rhs_color = Color(rhs);
    CHECK(lhs_color.has_value() && rhs_color.has_value());
    return *lhs_color == *rhs_color;
  }
  return partition_->SameMesh(lhs, rhs);
}

absl::StatusOr<bool> MpmdComputationFusion::Visit(HloComputation* computation) {
  bool changed = false;
  bool found_match = true;
  while (found_match) {
    // the fusion removes instructions/computations
    // and the fused tasks might have new tasks they could fuse with
    // for now, just do a naive N^2 loop that visits everything again
    // if a valid fusion was found in the
    absl::flat_hash_set<HloInstruction*> deleted;

    auto fuse = [&](HloInstruction* producer, HloInstruction* consumer) {
      VLOG(5) << producer->name() << ":"
              << producer->called_computations()[0]->name()
              << " has task with same color and same loop, "
                 "will fuse with "
              << consumer->name() << ":"
              << consumer->called_computations()[0]->name();
      // this producer is consumed by a single consumer, we can just fuse
      // them together to make a single task
      TF_RETURN_IF_ERROR(FuseComputations(computation, producer, consumer));
      changed = true;
      deleted.insert(producer);
      deleted.insert(consumer);

      return absl::OkStatus();
    };

    auto postorder = computation->MakeInstructionPostOrder();
    absl::flat_hash_map<HloInstruction*, int64_t> depths;
    std::vector<HloInstruction*> calls;
    for (auto* instruction : postorder) {
      int64_t instruction_depth = -1;
      for (auto* operand : instruction->operands()) {
        instruction_depth = std::max(instruction_depth, depths[operand]);
      }
      depths[instruction] = instruction_depth + 1;
      if (instruction->opcode() == HloOpcode::kCall) {
        calls.push_back(instruction);
      }
    }

    // first try to fuse with consumers of this instruction
    for (auto* instruction : calls) {
      if (deleted.contains(instruction)) {
        continue;
      }

      absl::InlinedVector<HloInstruction*, 4> consumers;
      for (auto* user : instruction->users()) {
        CHECK(user->opcode() == HloOpcode::kGetTupleElement);
        for (auto* maybe_call : user->users()) {
          if (maybe_call->opcode() == HloOpcode::kCall) {
            consumers.push_back(maybe_call);
          }
        }
      }

      std::stable_sort(consumers.begin(), consumers.end(),
                       [&](HloInstruction* lhs, HloInstruction* rhs) {
                         return depths[lhs] < depths[rhs];
                       });

      if (!consumers.empty()) {
        auto* candidate = consumers.front();
        int64_t min_user_depth = std::numeric_limits<int64_t>::max();
        for (auto* user : instruction->users()) {
          CHECK(user->opcode() == HloOpcode::kGetTupleElement);
          for (auto* call : user->users()) {
            if (call != candidate && call != computation->root_instruction()) {
              VLOG(5) << "for fusing " << instruction->name() << " user "
                      << call->name() << " of candidate " << candidate->name()
                      << " has depth " << depths[call];
              min_user_depth = std::min(min_user_depth, depths[call]);
            }
          }
        }
        VLOG(5) << instruction->name() << " requires depth " << min_user_depth
                << " from consumer " << candidate->name() << " to fuse"
                << ", actual depth=" << depths[candidate]
                << ", fusion match=" << std::boolalpha
                << FusionMatch(instruction, candidate)
                << ", fusion type=" << type_;
        if (depths[candidate] < min_user_depth &&
            FusionMatch(instruction, candidate)) {
          TF_RETURN_IF_ERROR(fuse(instruction, candidate));
        }
      }
    }

    // now fuse with any task that is later that matches, regardless of
    // whether they share a producer/consumer relationship
    for (int64_t lhs = 0; lhs < calls.size(); ++lhs) {
      HloInstruction* lhs_instruction = calls[lhs];
      if (deleted.contains(lhs_instruction)) {
        continue;
      }

      int64_t min_user_depth = std::numeric_limits<int64_t>::max();
      for (auto* user : lhs_instruction->users()) {
        CHECK(user->opcode() == HloOpcode::kGetTupleElement);
        for (auto* maybe_call : user->users()) {
          if (maybe_call->opcode() == HloOpcode::kCall ||
              (maybe_call->opcode() == HloOpcode::kTuple &&
               maybe_call != computation->root_instruction())) {
            min_user_depth = std::min(min_user_depth, depths[user]);
          }
        }
      }

      for (int64_t rhs = lhs + 1; rhs < calls.size(); ++rhs) {
        HloInstruction* rhs_instruction = calls[rhs];
        if (!deleted.contains(rhs_instruction) &&
            depths[rhs_instruction] < min_user_depth &&
            FusionMatch(rhs_instruction, lhs_instruction)) {
          TF_RETURN_IF_ERROR(fuse(lhs_instruction, rhs_instruction));
          break;
        }
      }
    }
    found_match = !deleted.empty();
  }

  return changed;
}

absl::StatusOr<bool> MpmdComputationFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(5) << "starting fusion pass for type=" << type_
          << " only_fuse_loop=" << std::boolalpha << only_fuse_loop_tasks_
          << " on module " << module->name();
  bool changed = false;
  std::vector<HloComputation*> to_visit = {module->entry_computation()};
  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        to_visit.push_back(instruction->called_computations()[0]);
      }
    }

    if (computation != module->entry_computation() || !only_fuse_loop_tasks_) {
      TF_ASSIGN_OR_RETURN(bool computation_changed, Visit(computation));
      changed |= computation_changed;
    }
  }

  return changed;
}

}  // namespace xla
