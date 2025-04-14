/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_sharding_propagation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "src/zuku/mesh.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/legate/legate_sharding.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"
#include "xla/service/sharding_propagation.h"
#include "xla/shape.h"
#include "xla/util.h"

namespace xla {

namespace {

constexpr int64_t kMinShardOverrideSize = 10 * 1024 * 1024;

// Returns true if the given `sharding` can be applied to the
// global `shape`. Returns false if the sharding extent is
// larger than the shape in a given dimension.
bool ShardingFits(const HloSharding& sharding, const Shape& shape) {
  if (sharding.IsReplicated()) {
    return true;
  }

  for (int64_t dim = 0; dim < shape.dimensions_size(); ++dim) {
    if (shape.dimensions(dim) < sharding.tile_assignment().dim(dim)) {
      return false;
    }
  }

  return true;
}

std::shared_ptr<const HloSharding> ElementwiseUsersShouldShardReplicated(
    const HloInstruction* instruction) {
  // already sharded
  if (instruction->has_sharding() && !instruction->sharding().IsReplicated()) {
    return nullptr;
  }
  std::shared_ptr<const HloSharding> candidate{nullptr};
  int64_t num_sharded_users = 0;
  for (auto* user : instruction->users()) {
    if (user->IsElementwise() && user->has_sharding()) {
      if (!user->sharding().IsReplicated()) {
        ++num_sharded_users;
        if (candidate) {
          if (*candidate != user->sharding()) {
            // different shardings in play, just keep replicated
            // since there is no one preferred sharding pattern
            return nullptr;
          }
        } else {
          candidate = user->sharding_ptr();
        }
      }
    }
  }

  return candidate;
}

bool ShardingCanBeReassigned(const HloInstruction* instruction) {
  return !instruction->has_sharding() || !HasAssignedAxes(instruction);
}

bool SourceHasImprovedSharding(const HloInstruction* source,
                               HloInstruction* target,
                               bool replicated_is_better_than_nothing) {
  if (source->sharding().IsReplicated()) {
    if (!target->has_sharding() && replicated_is_better_than_nothing) {
      return true;
    }
    return false;
  }

  if (!target->has_sharding() || target->sharding().IsReplicated()) {
    return true;
  }

  if (target->sharding().ReplicateOnLastTileDim()) {
    if (source->sharding().ReplicateOnLastTileDim()) {
      return target->sharding().tile_assignment().dimensions().back() <
             source->sharding().tile_assignment().dimensions().back();
    }
    // target has partial replication, sourcd does not
    return true;
  }
  // target is fully sharded
  return false;
}

void PropagateSharding(const HloInstruction* source, HloInstruction* target,
                       const InstructionProperties& properties,
                       bool propagate_replicated) {
  if (source->has_sharding() && ShardingCanBeReassigned(target)) {
    // parameters must always share their replicated sharding if assigned
    if (SourceHasImprovedSharding(source, target, propagate_replicated) ||
        properties.ParameterAlias(target)) {
      VLOG(5) << "propagating sharding from " << source->name() << " to "
              << target->name() << ": " << source->sharding() << " "
              << source->shape();
      target->set_sharding(source->sharding_ptr());
    }
  }
}

void PropagateSharding(const HloInstruction* source, HloInstruction* target,
                       int64_t num_devices,
                       const InstructionProperties& properties,
                       bool propagate_replicated) {
  PropagateSharding(source, target, properties, propagate_replicated);
  if (target->has_sharding() && !target->sharding().IsReplicated() &&
      target->sharding().tile_assignment().num_elements() != num_devices) {
    auto resized_sharding = ResizeSharding(target->shape(), target->sharding(),
                                           num_devices, std::nullopt);
    if (!resized_sharding.has_value()) {
      resized_sharding = HloSharding::Replicate();
      LOG(WARNING) << "unable to resize sharding " << source->sharding()
                   << " on " << source->name() << " to " << num_devices
                   << " on " << target->name()
                   << " elements, falling back to replicated";
    }
    VLOG(5) << "resized sharding to " << *resized_sharding << " for "
            << target->name() << " over " << num_devices << " devices from "
            << source->sharding() << " on " << source->name();

    target->set_sharding(*std::move(resized_sharding));
  }
}

void PropagateShardingToUsers(const HloInstruction* instruction,
                              const InstructionProperties& properties,
                              bool propagate_replicated) {
  for (auto* user : instruction->users()) {
    if (user->opcode() == HloOpcode::kCall) {
      const int64_t parameter_number = user->operand_index(instruction);
      auto* alias = user->called_computations()[0]->parameter_instruction(
          parameter_number);
      VLOG(5) << "maybe sharing sharding from input " << instruction->name()
              << " to param " << alias->name() << ": "
              << instruction->sharding_or_default(HloSharding::Replicate());
      PropagateSharding(instruction, alias, properties, propagate_replicated);
    }
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<HloModule>>
MpmdShardingPropagation::PropagateComputation(
    HloComputation* computation, HloInstruction* call,
    const InstructionProperties& properties, bool propagate_to_parameters) {
  const int64_t num_devices = partition_->NumDevicesForInstruction(call);

  HloModuleConfig subconfig = computation->parent()->config();
  absl::InlinedVector<bool, 1> allow_output_sharding_vec{true};
  subconfig.set_allow_spmd_sharding_propagation_to_output(
      allow_output_sharding_vec);

  absl::InlinedVector<bool, 8> allow_sharding_propagation_to_params(
      call->operand_count(), propagate_to_parameters);
  for (int64_t index = 0; index < call->operand_count(); ++index) {
    auto param_number = properties.ParameterNumber(call->operand(index));
    if (param_number.has_value()) {
      if (subconfig.allow_spmd_sharding_propagation_to_parameters().size() >
          *param_number) {
        allow_sharding_propagation_to_params[index] =
            subconfig
                .allow_spmd_sharding_propagation_to_parameters()[*param_number];
      } else if (subconfig.allow_spmd_sharding_propagation_to_parameters()
                     .empty()) {
        allow_sharding_propagation_to_params[index] = false;
      } else {
        allow_sharding_propagation_to_params[index] =
            subconfig.allow_spmd_sharding_propagation_to_parameters()[0];
      }
    }
  }

  subconfig.set_num_partitions(num_devices);
  subconfig.set_replica_count(1);
  auto module_to_propagate = std::make_unique<HloModule>(
      std::string(computation->name()), std::move(subconfig));

  HloCloneContext context{module_to_propagate.get()};
  module_to_propagate->AddComputation(computation->CloneInContext(context),
                                      /*is_entry=*/true);

  const bool is_spmd = num_devices > 1;
  ShardingPropagation spmd_propagation{
      is_spmd,
      /*propagate_metadata=*/false,
      /*allow_spmd_sharding_propagation_to_output=*/{true},
      /*allow_spmd_sharding_propagation_to_parameters=*/
      allow_sharding_propagation_to_params};

  TF_ASSIGN_OR_RETURN(bool smpd_changed,
                      spmd_propagation.Run(module_to_propagate.get()));

  return module_to_propagate;
}

void MpmdShardingPropagation::FixReshapeSharding(HloInstruction* call) {
  auto* comp = call->called_computations()[0];
  auto* parent_root = call->parent()->root_instruction();
  const bool root_has_explicit_sharding = parent_root->has_sharding();

  auto no_op_reshape_with_sharding = [](HloInstruction* i) {
    return i->opcode() == HloOpcode::kReshape &&
           !IsReplicatedOrNotSharded(i->operand(0)) &&
           IsReplicatedOrNotSharded(i) &&
           i->operand(0)->shape().dimensions() == i->shape().dimensions();
  };

  for (auto* instruction : comp->MakeInstructionPostOrder()) {
    if (no_op_reshape_with_sharding(instruction)) {
      instruction->set_sharding(instruction->operand(0)->sharding_ptr());
      VLOG(5) << "patching reshape " << instruction->name()
              << " to have sharding " << instruction->sharding() << " from "
              << instruction->operand(0)->name();
      if (instruction->users().size() == 1 &&
          instruction->users().front() == comp->root_instruction()) {
        comp->root_instruction()->clear_sharding();
        const int64_t operand_index =
            comp->root_instruction()->operand_index(instruction);
        for (auto* user : call->users()) {
          if (user->tuple_index() == operand_index) {
            VLOG(5) << "patching reshape user " << user->name()
                    << " to have sharding " << instruction->sharding();
            user->set_sharding(instruction->sharding_ptr());
            if (user->users().size() == 1 &&
                user->users().front() == parent_root &&
                !root_has_explicit_sharding) {
              // clear the sharding, let it be recomputed
              parent_root->clear_sharding();
            }
            break;
          }
        }
      }
    }
  }
}

absl::Status MpmdShardingPropagation::ReplaceCallWithShardedTwin(
    HloInstruction* call, HloComputation* sharded_computation,
    HloModule* module, InstructionProperties& properties,
    HloPassCleanup& cleanup) {
  HloCloneContext clone_context{module};
  auto* new_comp = module->AddComputationAndUnifyNamesAndIds(
      sharded_computation->CloneInContext(clone_context),
      /*is_entry=*/false);

  // compute the new properties for the replaced computation
  // and add them into the map
  properties.Add(InstructionProperties::Create(new_comp));

  HloInstruction* new_call = call->parent()->AddInstruction(
      HloInstruction::CreateCall(call->shape(), call->operands(), new_comp));

  auto* new_call_root = new_comp->root_instruction();
  for (auto* user : call->users()) {
    TF_RETURN_IF_ERROR(user->ReplaceOperandWith(0, new_call));
    auto* new_root_alias = new_call_root->mutable_operand(user->tuple_index());
    user->set_sharding(new_root_alias->sharding_ptr());
    properties.AddBackwardAlias(user, new_root_alias);
  }

  PropagateColor(call, new_call);

  FixReshapeSharding(new_call);

  HloComputation* unsharded_computation = call->called_computations()[0];
  cleanup.RemoveInstruction(call);
  cleanup.RemoveComputation(unsharded_computation);
  return absl::OkStatus();
}

absl::Status MpmdShardingPropagation::ShardForwardWhileInstruction(
    HloInstruction* instruction, InstructionProperties& properties,
    HloPassCleanup& cleanup) {
  const bool propagate_replicated = mode_ == PropagationMode::ForwardFull;

  auto* input_tuple = instruction->mutable_operand(0);
  auto* comp = instruction->called_computations()[0];
  auto* root = comp->root_instruction();
  auto* arg_tuple =
      instruction->called_computations()[0]->parameter_instruction(0);

  for (auto* arg : arg_tuple->users()) {
    auto* input = input_tuple->mutable_operand(arg->tuple_index());
    VLOG(5) << instruction->name() << " passes input " << input->name()
            << " to " << arg->name();
    PropagateSharding(input, arg, properties, propagate_replicated);
  }
  for (auto* user : instruction->users()) {
    auto* root_alias = root->mutable_operand(user->tuple_index());
    VLOG(5) << user->name() << " aliases root " << root_alias->name();
    PropagateSharding(user, root_alias, properties, propagate_replicated);
  }

  TF_RETURN_IF_ERROR(
      ShardForward(instruction->called_computations()[0], properties, cleanup));

  for (auto* arg : arg_tuple->users()) {
    auto* input = input_tuple->mutable_operand(arg->tuple_index());
    auto& arg_properties = properties.Get(arg);
    if (arg_properties.loop_carried_index.has_value()) {
      // loop carried
      auto* matching_root = root->mutable_operand(arg->tuple_index());
      properties.ForEachBackwardAliasAndSelf(arg, [&](HloInstruction* i) {
        VLOG(5) << "loop-carried " << matching_root->name()
                << " trying to match " << i->name();
        PropagateSharding(matching_root, i, properties, propagate_replicated);
      });
      PropagateShardingToUsers(arg, properties, propagate_replicated);
      properties.ForEachBackwardAliasAndSelf(input, [&](HloInstruction* i) {
        VLOG(5) << "loop-carried " << matching_root->name()
                << " trying to match " << i->name();
        PropagateSharding(matching_root, i, properties, propagate_replicated);
      });
      // also propagate sharding to the loop-carried initializer
      if (arg_properties.loop_carried_initializer == nullptr) {
        return InvalidArgumentStrCat("loop-carried initializer for ",
                                     matching_root->name(), " and ",
                                     arg->name(), " is null");
      }
      VLOG(5) << arg->name() << " propagating sharding from "
              << matching_root->name() << " to initializer "
              << arg_properties.loop_carried_initializer->name() << ": "
              << matching_root->sharding_or_default(HloSharding::Replicate());
      properties.ForEachBackwardAliasAndSelf(
          arg_properties.loop_carried_initializer, [&](HloInstruction* i) {
            PropagateSharding(matching_root, i, properties,
                              propagate_replicated);
          });
    }
  }

  for (auto* user : instruction->users()) {
    auto* root_alias = root->mutable_operand(user->tuple_index());
    VLOG(5) << "root alias " << root_alias->name() << " propagating to user "
            << user->name();
    PropagateSharding(root_alias, user, properties, propagate_replicated);
  }

  return absl::OkStatus();
}

absl::Status MpmdShardingPropagation::ShardForwardCallInstruction(
    HloInstruction* instruction, InstructionProperties& properties,
    HloPassCleanup& cleanup) {
  auto* computation = instruction->called_computations()[0];
  auto task_color = Color(instruction);
  if (!task_color.has_value()) {
    return InvalidArgumentStrCat(instruction->name(),
                                 " was not assigned a color");
  }

  const bool propagate_replicated = mode_ == PropagationMode::ForwardFull;
  const bool propagate_to_parameters =
      mode_ == PropagationMode::ForwardInputOutput;

  zuku::DeviceList devices = partition_->DefaultPerTaskMesh(*task_color);
  for (int64_t index = 0; index < instruction->operand_count(); ++index) {
    auto* input = instruction->operand(index);
    auto* param = computation->parameter_instruction(index);
    auto param_color = Color(param);
    if (!param_color.has_value() || *param_color != *task_color) {
      return InvalidArgumentStrCat("parameter ", param->name(), " has color ",
                                   param_color.value_or("unassigned"),
                                   ", but expected ", *task_color);
    }
    VLOG(5) << input->name() << " passing to parameter " << param->name()
            << ": " << input->sharding_or_default(HloSharding::Replicate());
    PropagateSharding(input, param, devices.size(), properties,
                      propagate_replicated);
    VLOG(5) << param->name() << " now has sharding "
            << param->sharding_or_default(HloSharding::Replicate());
  }

  // XLA does not propagate sharding to dynamic slices or reshape in the
  // expected way
  for (auto* param : computation->parameter_instructions()) {
    for (auto* user : param->users()) {
      if (user->opcode() == HloOpcode::kDynamicSlice && param->has_sharding() &&
          ShardingFits(param->sharding(), user->shape())) {
        PropagateSharding(param, user, properties, propagate_replicated);
      }
    }
  }

  auto* parent_root = instruction->parent()->root_instruction();

  FixReshapeSharding(instruction);

  TF_ASSIGN_OR_RETURN(auto sharded_twin_module,
                      PropagateComputation(computation, instruction, properties,
                                           propagate_to_parameters));

  if (shard_iota_) {
    for (auto* instruction :
         sharded_twin_module->entry_computation()->instructions()) {
      if (instruction->opcode() == HloOpcode::kIota &&
          !instruction->has_sharding()) {
        const int64_t size = instruction->shape().dimensions(0);
        if (size >= devices.size() && size % devices.size() == 0) {
          instruction->set_sharding(HloSharding::IotaTile({devices.size()}));
        }
      }
    }
  }

  for (auto* subcomp : sharded_twin_module->computations()) {
    for (auto* instruction : subcomp->instructions()) {
      if (instruction->has_sharding() && !instruction->sharding().IsTuple() &&
          !instruction->sharding().IsReplicated()) {
        if (instruction->sharding().tile_assignment().num_elements() !=
            devices.size()) {
          LOG(FATAL) << instruction->name() << " " << instruction->sharding()
                     << " " << devices;
        }
      }
    }
  }

  // TODO: XLA awfulness
  // XLA effs up optimization barriers, remove all sharding from them to avoid
  // the XLA bug
  for (auto* instruction :
       sharded_twin_module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kOptimizationBarrier &&
        instruction->has_sharding()) {
      instruction->clear_sharding();
    }
  }

  const int64_t num_instructions = computation->instruction_count();
  if (sharded_twin_module->entry_computation()->instruction_count() !=
      num_instructions) {
    return InvalidArgumentStrCat(
        "computation ", computation->name(),
        " has different no. of instructions after sharding propagation");
  }

  if (mode_ == PropagationMode::ForwardFull) {
    // replace the original computation with the fully sharded computation
    TF_RETURN_IF_ERROR(ReplaceCallWithShardedTwin(
        instruction, sharded_twin_module->entry_computation(),
        computation->parent(), properties, cleanup));
  } else {
    PropagateOutputShardingsForward(sharded_twin_module->entry_computation(),
                                    instruction, properties);
    PropagateParamShardingsBackward(sharded_twin_module->entry_computation(),
                                    instruction, properties);
  }

  return absl::OkStatus();
}

MpmdShardingPropagation::MpmdShardingPropagation(
    HloPartition* partition, PropagationMode mode,
    MpmdShardingPropagationConfig cfg)
    : partition_(partition),
      shard_iota_(cfg.shard_iota),
      mode_(mode),
      min_shard_override_size_(
          cfg.min_shard_override_size.value_or(kMinShardOverrideSize)) {}

void MpmdShardingPropagation::PropagateParamShardingsBackward(
    HloComputation* sharded_computation, HloInstruction* call_instruction,
    const InstructionProperties& properties) {
  for (int64_t index = 0; index < call_instruction->operand_count(); ++index) {
    auto* param = sharded_computation->parameter_instruction(index);
    auto* operand = call_instruction->mutable_operand(index);
    if (!IsReplicatedOrNotSharded(param)) {
      properties.ForEachBackwardAliasAndSelf(operand, [&](HloInstruction* i) {
        if (!i->has_sharding()) {
          VLOG(5) << "propagate backward from param " << param->name()
                  << " to input " << i->name() << ": "
                  << param->sharding_or_default(HloSharding::Replicate());
          PropagateSharding(param, i, partition_->NumDevicesForInstruction(i),
                            properties,
                            /*propagate_replicated=*/false);
        }
      });
    }
  }
}

void MpmdShardingPropagation::PropagateOutputShardingsForward(
    HloComputation* sharded_computation, HloInstruction* call_instruction,
    const InstructionProperties& properties) {
  VLOG(7) << sharded_computation->ToString();
  auto* call_root = sharded_computation->root_instruction();
  for (auto* user : call_instruction->users()) {
    auto* alias = call_root->mutable_operand(user->tuple_index());
    if (!user->has_sharding() && !IsReplicatedOrNotSharded(alias)) {
      VLOG(5) << "propagate forward from root " << alias->name() << " to user "
              << user->name() << " " << user->shape() << ": "
              << alias->sharding();
      user->set_sharding(alias->sharding_ptr());
    }
  }
}

absl::Status MpmdShardingPropagation::ShardBackward(
    HloComputation* computation, const InstructionProperties& properties,
    HloPassCleanup& cleanup) {
  auto postorder = computation->MakeInstructionPostOrder();
  for (auto iter = postorder.rbegin(); iter != postorder.rend(); ++iter) {
    HloInstruction* instruction = *iter;
    if (instruction->opcode() == HloOpcode::kWhile) {
      TF_RETURN_IF_ERROR(ShardBackward(instruction->called_computations()[0],
                                       properties, cleanup));
    } else if (instruction->opcode() == HloOpcode::kCall) {
      TF_ASSIGN_OR_RETURN(
          auto sharded_twin_module,
          PropagateComputation(instruction->called_computations()[0],
                               instruction, properties,
                               /*propagate_to_parameters=*/true));
      HloComputation* twin_computation =
          sharded_twin_module->entry_computation();

      PropagateParamShardingsBackward(twin_computation, instruction,
                                      properties);
      PropagateOutputShardingsForward(twin_computation, instruction,
                                      properties);
    }
  }
  return absl::OkStatus();
}

absl::Status MpmdShardingPropagation::ShardForward(
    HloComputation* computation, InstructionProperties& properties,
    HloPassCleanup& cleanup) {
  // the root tuple might have shardings that are not propagated to their inputs
  auto* root = computation->root_instruction();

  struct LoopCarriedInstruction {
    zuku::DeviceList devices;
    HloInstruction* instruction;
  };
  absl::flat_hash_map<int64_t, absl::InlinedVector<LoopCarriedInstruction, 3>>
      loop_carried_instructions;

  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    switch (instruction->opcode()) {
      case HloOpcode::kCall: {
        TF_RETURN_IF_ERROR(
            ShardForwardCallInstruction(instruction, properties, cleanup));
        break;
      }
      case HloOpcode::kWhile: {
        TF_RETURN_IF_ERROR(
            ShardForwardWhileInstruction(instruction, properties, cleanup));
        break;
      }
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kTuple:
      case HloOpcode::kGetTupleElement:
        // this do-nothing case is here for sanity checking
        // these are valid instructions to be visiting
        break;
      default:
        if (instruction->opcode() != HloOpcode::kCustomCall) {
          return InvalidArgumentStrCat(
              computation->parent()->name(),
              " is not in canonical MPMD form, must "
              "consist only of parameters, calls, and custom-calls: found ",
              instruction->ToString());
        }
        break;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> MpmdShardingPropagation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  HloInstruction* root = module->entry_computation()->root_instruction();
  ApplyRootTupleShardingsToOperands(module, root);

  auto properties = InstructionProperties::Create(module);
  HloPassCleanup cleanup;
  switch (mode_) {
    case PropagationMode::ForwardInputOutput:
    case PropagationMode::ForwardFull:
      TF_RETURN_IF_ERROR(
          ShardForward(module->entry_computation(), properties, cleanup));
      break;
    case PropagationMode::BackwardInputOutput:
      TF_RETURN_IF_ERROR(
          ShardBackward(module->entry_computation(), properties, cleanup));
      break;
  }

  TF_RETURN_IF_ERROR(cleanup.CleanUp());
  return true;
}

}  // namespace xla
