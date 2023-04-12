/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_shard_map_loop_reduce.h"

#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/mm_sharding.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/service/all_reduce_folder.h"
#include "xla/service/gpu/transforms/all_reduce_splitter.h"
#include "xla/service/gpu/transforms/reduce_scatter_creator.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"

namespace xla {
namespace {

bool IsAddComputation(const HloComputation* comp) {
  HloInstruction* root = comp->root_instruction();
  return root->opcode() == HloOpcode::kAdd &&
         root->operand(0) == comp->parameter_instruction(0) &&
         root->operand(1) == comp->parameter_instruction(1);
}

bool ShardedOverCxnDims(const HloInstruction* i) {
  if (i->opcode() == HloOpcode::kReduce &&
      i->called_computations()[0]->root_instruction()->opcode() ==
          HloOpcode::kAdd) {
    auto* reduce = static_cast<const HloReduceInstruction*>(i);
    auto* operand = i->operand(0);
    if (!IsReplicatedOrNotSharded(operand)) {
      for (int64_t dim : reduce->dimensions()) {
        if (operand->sharding().tile_assignment().dim(dim) > 1) {
          return true;
        }
      }
    }
  } else if (i->opcode() == HloOpcode::kDot) {
    auto* dot = static_cast<const HloDotInstruction*>(i);
    auto* lhs = i->operand(0);
    if (!IsReplicatedOrNotSharded(lhs)) {
      for (int64_t dim :
           dot->dot_dimension_numbers().lhs_contracting_dimensions()) {
        if (lhs->sharding().tile_assignment().dim(dim) > 1) {
          return true;
        }
      }
    }

    auto* rhs = i->operand(1);
    if (!IsReplicatedOrNotSharded(rhs)) {
      for (int64_t dim :
           dot->dot_dimension_numbers().rhs_contracting_dimensions()) {
        if (rhs->sharding().tile_assignment().dim(dim) > 1) {
          return true;
        }
      }
    }
  } else if (i->opcode() == HloOpcode::kScatter) {
    auto* update_comp = i->called_computations()[0];
    if (!IsAddComputation(update_comp)) {
      return false;
    }

    // check that to_apply is an add and that the sharding is non-trivial
    auto* scatter = static_cast<const HloScatterInstruction*>(i);
    absl::Span<HloInstruction* const> operands = scatter->scatter_operands();
    absl::Span<HloInstruction* const> updates = scatter->scatter_updates();

    for (int i = 0; i < operands.length(); i++) {
      HloInstruction* operand = operands[i];
      HloInstruction* update = updates[i];

      if (!IsReplicatedOrNotSharded(operand) &&
          !IsReplicatedOrNotSharded(update) &&
          operand->sharding().ReplicateOnLastTileDim() &&
          !update->sharding().ReplicateOnLastTileDim()) {
        // partial updates are sent to the operand so an
        // all-reduce must be performed to complete sum
        return true;
      }
    }
  }

  return false;
}

}  // namespace

absl::StatusOr<std::unique_ptr<HloModule>>
MpmdShardMapLoopReduce::CreateOutlinedPartitionedModule(
    HloInstruction* instruction) {
  HloModuleConfig subconfig = instruction->parent()->parent()->config();
  absl::InlinedVector<bool, 1> allow_output_sharding_vec{true};
  subconfig.set_allow_spmd_sharding_propagation_to_output(
      allow_output_sharding_vec);
  absl::InlinedVector<bool, 1> allow_param_sharding_vec{false};
  // subconfig.set_allow_spmd_sharding_propagation_to_parameters(allow_param_sharding_vec);

  const int64_t num_partitions =
      instruction->operand(0)->sharding().tile_assignment().num_elements();

  subconfig.set_num_partitions(num_partitions);
  subconfig.set_replica_count(1);
  subconfig.set_use_auto_spmd_partitioning(false);
  HloComputation::Builder builder{instruction->name()};

  auto outlined_module = std::make_unique<HloModule>(
      std::string(instruction->name()), std::move(subconfig));
  HloCloneContext outlined_context{outlined_module.get()};

  absl::InlinedVector<HloInstruction*, 3> operands;
  int64_t param_number = 0;
  for (auto* operand : instruction->operands()) {
    TF_ASSIGN_OR_RETURN(
        auto* new_operand,
        builder.AddParameter(HloInstruction::CreateParameter(
            param_number++, operand->shape(), std::string(operand->name()))));
    operands.push_back(new_operand);
    new_operand->set_sharding(operand->sharding_ptr());
  }

  auto* outlined_root =
      builder.AddInstruction(instruction->CloneWithNewOperands(
          instruction->shape(), operands, &outlined_context));
  HloComputation* computation =
      outlined_module->AddComputationAndUnifyNamesAndIds(
          builder.Build(outlined_root), /*is_entry=*/true);

  spmd::StatefulRngSpmdPartitioner partitioner{num_partitions, 1, {}};
  TF_ASSIGN_OR_RETURN(bool partitioned, partitioner.Run(outlined_module.get()));

  AllReduceFolder folder{};
  TF_ASSIGN_OR_RETURN(bool folded, folder.Run(outlined_module.get()));

  AllReduceSplitter splitter{/*always_split=*/true};
  TF_ASSIGN_OR_RETURN(bool split, splitter.Run(outlined_module.get()));

  gpu::ReduceScatterCreator creator{};
  TF_ASSIGN_OR_RETURN(bool created_rs, creator.Run(outlined_module.get()));

  return std::move(outlined_module);
}

absl::Status MpmdShardMapLoopReduce::ReplaceWithOutlinedPartitionedModule(
    HloInstruction* instruction, const HloModule& module,
    HloPassCleanup& cleanup) {
  HloCloneContext context{instruction->parent()->parent()};

  // this all-reduce can be lifted out of the loop
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> clone_map;
  auto color = Color(instruction);
  for (auto* outlined :
       module.entry_computation()->MakeInstructionPostOrder()) {
    if (outlined->opcode() == HloOpcode::kParameter) {
      auto* operand =
          instruction->mutable_operand(outlined->parameter_number());
      auto* manual_shard = instruction->parent()->AddInstruction(
          HloInstruction::CreateCustomCall(GetSpmdShape(operand), {operand},
                                           "SPMDFullToShardShape"));
      manual_shard->set_sharding(HloSharding::Manual());
      clone_map[outlined] = manual_shard;
      if (color.has_value()) {
        AssignColor(manual_shard, *color);
      }
    } else {
      absl::InlinedVector<HloInstruction*, 3> new_operands;
      for (auto* operand : outlined->operands()) {
        auto iter = clone_map.find(operand);
        if (iter == clone_map.end()) {
          return InvalidArgumentStrCat("operand ", operand->name(),
                                       " not found in outlined clone map for ",
                                       instruction->name());
        }
        new_operands.push_back(iter->second);
      }
      auto* clone =
          instruction->parent()->AddInstruction(outlined->CloneWithNewOperands(
              outlined->shape(), new_operands, &context));
      clone_map[outlined] = clone;
      if (color.has_value()) {
        AssignColor(clone, *color);
      }
    }
  }

  auto iter = clone_map.find(module.entry_computation()->root_instruction());
  if (iter == clone_map.end()) {
    return InvalidArgumentStrCat("outlined root not found while shard mapping ",
                                 instruction->name());
  }
  auto* clone = iter->second;
  if (clone->opcode() == HloOpcode::kTuple) {
    // we can replace all the get-tuple-element users directly with the
    // partitionined instruction
    for (auto* user : instruction->users()) {
      auto* partitioned_tuple_input =
          clone->mutable_operand(user->tuple_index());
      auto* sharded_output = instruction->parent()->AddInstruction(
          HloInstruction::CreateCustomCall(user->shape(),
                                           {partitioned_tuple_input},
                                           "SPMDShardToFullShape"));
      sharded_output->set_sharding(user->sharding_ptr());
      partitioned_tuple_input->set_sharding(HloSharding::Manual());
      TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(sharded_output));
      cleanup.RemoveInstruction(user);
      if (color.has_value()) {
        AssignColor(sharded_output, *color);
      }
    }
    cleanup.RemoveInstruction(clone);
    cleanup.RemoveInstruction(instruction);
  } else {
    auto* sharded_output =
        instruction->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            instruction->shape(), {clone}, "SPMDShardToFullShape"));
    sharded_output->set_sharding(instruction->sharding_ptr());
    if (color.has_value()) {
      AssignColor(sharded_output, *std::move(color));
    }
    TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(sharded_output));
    cleanup.RemoveInstruction(instruction);
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> MpmdShardMapLoopReduce::VisitLoop(
    HloInstruction* loop, const InstructionProperties& properties,
    HloPassCleanup& cleanup) {
  bool changed = false;
  HloComputation* computation = loop->called_computations()[0];
  HloInstruction* root = computation->root_instruction();

  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    VLOG(5) << "visiting " << instruction->name();
    if ((instruction->opcode() == HloOpcode::kDot ||
         instruction->opcode() == HloOpcode::kReduce ||
         instruction->opcode() == HloOpcode::kScatter) &&
        ShardedOverCxnDims(instruction)) {
      VLOG(5) << instruction->name()
              << " is loop-carried add that is sharded over cxn dims in "
              << loop->name();
      absl::flat_hash_map<const HloInstruction*, int64_t> additive_to_outputs;
      for (auto* user : instruction->users()) {
        const auto& user_properties = properties.Get(user);
        if (user_properties.additive_to_while_input.empty()) {
          additive_to_outputs.clear();
          break;
        }

        for (auto* input : user_properties.additive_to_while_input) {
          additive_to_outputs[root->operand(input->tuple_index())] =
              input->tuple_index();
        }
      }

      if (additive_to_outputs.empty()) {
        VLOG(5) << instruction->name() << " is not additive to any outputs";
        continue;
      }

      const auto& pair = *additive_to_outputs.begin();
      VLOG(5) << instruction->name() << " is additive to output "
              << pair.first->name() << " at index " << pair.second;

      TF_ASSIGN_OR_RETURN(auto outlined_module,
                          CreateOutlinedPartitionedModule(instruction));

      auto* outlined_output =
          outlined_module->entry_computation()->root_instruction();

      if (outlined_output->opcode() == HloOpcode::kAllReduce ||
          outlined_output->opcode() == HloOpcode::kReduceScatter) {
        VLOG(5) << "replacing " << instruction->name()
                << " with shard-mapped collective";
        TF_RETURN_IF_ERROR(ReplaceWithOutlinedPartitionedModule(
            instruction, *outlined_module, cleanup));
        changed = true;
      } else {
        VLOG(5) << "not replacing " << instruction->name()
                << ", outlined output is " << outlined_output->ToString();
      }
    } else if (instruction->IsCustomCall("CustomSPMDPartitioning")) {
      // we have no visibility into what this is so we have no choice but to do
      // the partitioning now and replace the instruction with its partitioning
      TF_ASSIGN_OR_RETURN(auto outlined_module,
                          CreateOutlinedPartitionedModule(instruction));
      TF_RETURN_IF_ERROR(ReplaceWithOutlinedPartitionedModule(
          instruction, *outlined_module, cleanup));
      changed = true;
    }
  }
  return changed;
}

absl::StatusOr<bool> MpmdShardMapLoopReduce::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  auto properties = InstructionProperties::Create(module);
  HloPassCleanup cleanup;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      TF_ASSIGN_OR_RETURN(bool instruction_changed,
                          VisitLoop(instruction, properties, cleanup));
      changed |= instruction_changed;
    }
  }

  if (changed) {
    // go through and make sure every collective has a unique channel ID
    absl::flat_hash_set<int64_t> used_channel_ids;
    std::vector<HloComputation*> to_visit{module->entry_computation()};
    int64_t next_available = 1;
    while (!to_visit.empty()) {
      HloComputation* computation = to_visit.back();
      to_visit.pop_back();
      for (auto* instruction : computation->MakeInstructionPostOrder()) {
        switch (instruction->opcode()) {
          case HloOpcode::kAllReduce:
          case HloOpcode::kAllGather:
          case HloOpcode::kReduceScatter: {
            auto* channel_instruction =
                dynamic_cast<HloChannelInstruction*>(instruction);
            if (channel_instruction == nullptr) {
              return InvalidArgumentStrCat(
                  "collective did not cast to HloChannelInstruction");
            }
            if (channel_instruction->channel_id().has_value()) {
              if (used_channel_ids.contains(
                      channel_instruction->channel_id().value())) {
                while (used_channel_ids.contains(next_available)) {
                  next_available++;
                }
                channel_instruction->set_channel_id(next_available);
                ++next_available;
              }
              used_channel_ids.insert(*channel_instruction->channel_id());
            }
            break;
          }
          default: {
            for (auto* comp : instruction->called_computations()) {
              to_visit.push_back(comp);
            }
            break;
          }
        }
      }
    }
  }
  TF_RETURN_IF_ERROR(cleanup.CleanUp());
  return changed;
}

}  // namespace xla
