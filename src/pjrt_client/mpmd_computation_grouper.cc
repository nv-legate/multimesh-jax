/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_computation_grouper.h"

#include <limits>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

namespace {

struct BuildContext {
  std::string name;
  int64_t depth{0};
  std::string color{"none"};
  absl::flat_hash_map<HloInstruction*, HloInstruction*> clones;
  std::optional<HloComputation::Builder> builder;
  absl::InlinedVector<BuildContext*, 6> producers;
  absl::InlinedVector<BuildContext*, 6> consumers;
  std::vector<HloInstruction*> intermediate_roots;
  std::vector<HloInstruction*> intermediate_parameters;
  std::vector<HloInstruction*> parameters;
  std::vector<HloInstruction*> roots;
  absl::flat_hash_set<HloInstruction*> roots_added;
  HloInstruction* call_instruction{nullptr};
  HloComputation* call_to_recurse{nullptr};
};

bool IsMicrobatchLoop(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kWhile &&
         instruction->has_backend_config();
}

bool IsAliasedGetTupleElement(HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kGetTupleElement) {
    auto* operand = instruction->operand(0);
    if (operand->opcode() == HloOpcode::kParameter ||
        IsMicrobatchLoop(operand)) {
      return true;
    }
  }
  return false;
  auto* input_tuple = instruction->operand(0);
  if (IsAssignedColor(input_tuple)) {
    return false;
  }
  return input_tuple->opcode() == HloOpcode::kOptimizationBarrier ||
         input_tuple->opcode() == HloOpcode::kTuple ||
         input_tuple->opcode() == HloOpcode::kParameter ||
         input_tuple->opcode() == HloOpcode::kWhile;
  return false;
}

bool SkipInComputation(HloInstruction* instruction,
                       const InstructionProperties& properties) {
  if (instruction->opcode() == HloOpcode::kParameter ||
      properties.DerivedInput(instruction)) {
    return true;
  }

  // skip the input tuple to an unrolled while loop
  if (instruction->opcode() == HloOpcode::kTuple &&
      instruction->users().size() == 1 &&
      IsMicrobatchLoop(instruction->users().front())) {
    return true;
  }

  if (IsAliasedGetTupleElement(instruction)) {
    return true;
  }

  if (instruction->IsCustomCall("SliceOffset") ||
      instruction->IsCustomCall("DummyLoopOperation")) {
    return true;
  }

  return !IsAssignedColor(instruction);
}

bool ReplicateToUser(BuildContext* operand_context, HloInstruction* operand,
                     HloInstruction* instruction,
                     const InstructionProperties& properties) {
  // add as an operand
  if (operand->opcode() == HloOpcode::kParameter ||
      operand->opcode() == HloOpcode::kWhile ||
      !properties.AllowRecomputation(operand) ||
      operand->IsCustomCall("SliceOffset")) {
    return false;
  }

  if (operand->opcode() == HloOpcode::kOptimizationBarrier ||
      operand->opcode() == HloOpcode::kTuple ||
      operand->IsCustomCall("Sharding")) {
    // yes, pull these in and replicate wherever used
    return true;
  }

  if (operand->opcode() == HloOpcode::kGetTupleElement) {
    switch (operand->operand(0)->opcode()) {
      case HloOpcode::kOptimizationBarrier:
      case HloOpcode::kTuple:
        return !IsAssignedColor(operand->operand(0));
      case HloOpcode::kParameter:
        // get-tuple-element of arg tuple should be used directly
        return false;
      case HloOpcode::kWhile:
        // legate managed loops should not have their outputs replicated
        return !operand->operand(0)->has_backend_config();
      default:
        break;
    }
  }

  if (properties.DerivedInput(operand)) {
    return true;
  }

  // broadcasts increase the size of operands - so really we want to depend
  // on the operand to the broadcast
  return operand_context == nullptr ||
         operand->opcode() == HloOpcode::kBroadcast;
}

bool IsRoot(HloInstruction* instruction, HloComputation* comp) {
  if (comp->root_instruction()->shape().IsTuple()) {
    for (auto* user : instruction->users()) {
      if (user == comp->root_instruction()) {
        return true;
      }
    }
    return false;
  }
  return instruction == comp->root_instruction();
}

BuildContext* GetBuilder(
    HloInstruction* instruction,
    absl::flat_hash_map<HloInstruction*, BuildContext*>& builder_assignments) {
  auto iter = builder_assignments.find(instruction);
  if (iter == builder_assignments.end()) {
    return nullptr;
  }
  return iter->second;
}

absl::StatusOr<HloInstruction*> AddCloneToContext(
    HloComputation* computation, BuildContext* context,
    HloInstruction* instruction, HloPartition* partition,
    absl::flat_hash_map<HloInstruction*, BuildContext*>& builder_assignments,
    InstructionProperties& properties) {
  absl::InlinedVector<HloInstruction*, 6> new_operands;
  VLOG(5) << "adding " << instruction->name()
          << " to context for color=" << context->color
          << ",depth=" << context->depth;
  for (auto* operand : instruction->mutable_operands()) {
    BuildContext* operand_context = GetBuilder(operand, builder_assignments);
    VLOG(5) << instruction->name() << " has operand " << operand->name()
            << " on context " << operand_context;
    if (ReplicateToUser(operand_context, operand, instruction, properties)) {
      VLOG(5) << instruction->name() << " replicating operand "
              << operand->name();
      // see if this has already been cloned or this is a replicated instruction
      auto iter = context->clones.find(operand);
      if (iter != context->clones.end()) {
        new_operands.push_back(iter->second);
      } else {
        // we have to clone the entire tree backwards
        TF_ASSIGN_OR_RETURN(
            auto* clone,
            AddCloneToContext(computation, context, operand, partition,
                              builder_assignments, properties));

        new_operands.push_back(clone);
      }
    } else if (operand_context == context) {
      VLOG(5) << instruction->name() << " adding operand " << operand->name()
              << " to same context";
      new_operands.push_back(context->clones[operand]);
    } else {
      auto iter = context->clones.find(operand);
      if (iter == context->clones.end()) {
        TF_ASSIGN_OR_RETURN(
            auto* clone,
            context->builder->AddParameter(HloInstruction::CreateParameter(
                context->parameters.size(), operand->shape(),
                operand->name())));

        if (operand->has_sharding()) {
          // pass on replication to the cloned operand
          clone->set_sharding(operand->sharding_ptr());
        }

        AssignColor(clone, context->color);
        PropagateAxes(operand, clone);

        VLOG(5) << instruction->name() << " cloned operand " << operand->name()
                << " to " << clone->name() << ",color=" << context->color
                << " as parameter number " << context->parameters.size()
                << " on context " << context->name << ":" << context
                << " from context "
                << (operand_context ? operand_context->name : "null");

        context->intermediate_parameters.push_back(operand);
        context->parameters.push_back(operand);
        context->clones[operand] = clone;
        new_operands.push_back(clone);

        if (operand_context &&
            !operand_context->roots_added.contains(operand)) {
          VLOG(5) << operand->name()
                  << " is a temp output for color=" << operand_context->color
                  << ", context=" << operand_context;
          operand_context->roots_added.insert(operand);
          operand_context->intermediate_roots.push_back(operand);
          operand_context->roots.push_back(operand);
        }
      } else {
        auto* clone = iter->second;
        new_operands.push_back(clone);
      }
    }
  }

  auto* clone = context->builder->AddInstruction(
      instruction->CloneWithNewOperands(instruction->shape(), new_operands));

  VLOG(5) << instruction->name() << " cloned to " << clone->name()
          << " on context " << context->name << ":" << context->color;

  properties.Clone(instruction, clone);

  AssignColor(clone, context->color);
  PropagateAxes(instruction, clone);

  context->clones[instruction] = clone;
  if (IsRoot(instruction, computation)) {
    context->roots.push_back(instruction);
    context->roots_added.insert(instruction);
  }
  return clone;
};

bool InstructionCanGroupWithOperand(BuildContext* operand_context,
                                    HloInstruction* instruction,
                                    HloInstruction* operand,
                                    const InstructionProperties& properties) {
  if (IsMicrobatchLoop(operand) || IsMicrobatchLoop(instruction)) {
    return false;
  }

  auto instruction_color = Color(instruction);
  if (instruction_color.has_value() && operand_context != nullptr &&
      operand_context->color != *instruction_color) {
    VLOG(5) << instruction->name() << " cannot group with operand "
            << operand->name()
            << " due to different coloring: " << operand_context->color
            << " != " << *instruction_color;
    return false;
  }

  return true;
}

}  // namespace

absl::Status MpmdComputationGrouper::GroupComputationIntoTasks(
    HloComputation* computation, InstructionProperties& properties,
    HloInstruction* parent_call) {
  std::vector<std::unique_ptr<BuildContext>> builders;
  absl::flat_hash_map<HloInstruction*, BuildContext*> builder_assignments;
  absl::flat_hash_map<std::string, std::vector<BuildContext*>>
      builders_for_color;
  absl::flat_hash_map<HloInstruction*, int64_t> instruction_context_depth;
  auto get_builder = [&](HloInstruction* instruction) {
    auto iter = builder_assignments.find(instruction);
    if (iter == builder_assignments.end()) {
      return (BuildContext*)nullptr;
    }
    return iter->second;
  };

  absl::flat_hash_map<HloInstruction*, HloInstruction*> gte_root_clones;
  auto postorder = computation->MakeInstructionPostOrder();
  for (auto* instruction : postorder) {
    // the instruction is its own input
    if (instruction->IsCustomCall("SliceOffset")) {
      gte_root_clones[instruction] = instruction;
    } else if (instruction->opcode() == HloOpcode::kParameter &&
               instruction->shape().IsTuple()) {
      // all get-tuple-elements of an arg tuple should be used directly as task
      // inputs
      for (auto* user : instruction->users()) {
        gte_root_clones[user] = user;
      }
    }

    int64_t min_context_depth = 0;
    for (auto* operand : instruction->operands()) {
      if (!InstructionCanGroupWithOperand(get_builder(operand), instruction,
                                          operand, properties)) {
        min_context_depth =
            std::max(min_context_depth, instruction_context_depth[operand] + 1);
      } else {
        min_context_depth =
            std::max(min_context_depth, instruction_context_depth[operand]);
      }
      VLOG(5) << instruction->name() << " context depth is "
              << min_context_depth << " after operand " << operand->name();
    }
    instruction_context_depth[instruction] = min_context_depth;

    // this is a Legate-managed while loop
    if (IsMicrobatchLoop(instruction)) {
      builders.push_back(std::make_unique<BuildContext>(BuildContext{
          .name = std::string(instruction->name()),
          .depth = min_context_depth,
          .call_instruction = instruction,
          .call_to_recurse = instruction->called_computations()[0]}));

      continue;
    }

    if (SkipInComputation(instruction, properties)) {
      VLOG(5) << "skipping " << instruction->name();
      continue;
    }

    if (instruction->opcode() == HloOpcode::kParameter) {
      return InvalidArgumentStrCat(
          "parameter ", instruction->name(),
          " being added directly to a computation group");
    }

    // do not make any changes with a root tuple
    if (instruction == computation->root_instruction() &&
        instruction->shape().IsTuple()) {
      continue;
    }

    auto color = Color(instruction);
    if (!color.has_value()) {
      return InvalidArgumentStrCat(
          "visiting instruction ", instruction->name(),
          " without a color in MPMD computation group");
    }

    VLOG(5) << "visiting " << instruction->name() << " with color=" << *color
            << ",min_context_depth=" << min_context_depth;

    BuildContext* context_to_use = nullptr;

    VLOG(5) << instruction->name() << " final depth is " << min_context_depth;
    for (auto* context : builders_for_color[*color]) {
      if (context->depth < min_context_depth) {
        continue;
      }

      context_to_use = context;
      // this might get moved to a lower depth so update
      VLOG(5) << instruction->name() << " assigned to existing context "
              << context->name << ":" << context << " at depth "
              << context->depth;
      instruction_context_depth[instruction] = context->depth;
      break;
    }

    if (context_to_use == nullptr) {
      std::string name = *color;
      if (parent_call && parent_call->opcode() == HloOpcode::kWhile) {
        absl::StrAppend(&name, "_loop");
      }
      builders.push_back(std::make_unique<BuildContext>(BuildContext{
          .name = name,
          .depth = min_context_depth,
          .color = *color,
          .builder = HloComputation::Builder{name},
      }));
      context_to_use = builders.back().get();
      builders_for_color[*color].push_back(context_to_use);
      VLOG(5) << instruction->name() << " creating new context " << name
              << " for color=" << *color << ", depth=" << context_to_use->depth
              << ":  " << context_to_use;
    } else {
      VLOG(5) << instruction->name() << " using existing context "
              << context_to_use->name << " for color=" << *color
              << ", depth=" << context_to_use->depth << ":  " << context_to_use;
    }

    builder_assignments[instruction] = context_to_use;
  }

  // go back through in reverse order and try to move instructions as LATE as
  // possible
  for (auto iter = postorder.rbegin(); iter != postorder.rend(); ++iter) {
    HloInstruction* instruction = *iter;
    if (instruction == computation->root_instruction()) {
      continue;
    }

    auto* builder = get_builder(instruction);
    if (builder == nullptr) {
      continue;
    }

    int64_t min_user_depth = std::numeric_limits<int64_t>::max();
    bool all_users_assigned = true;
    BuildContext* min_depth_user_builder = nullptr;
    for (auto* user : instruction->users()) {
      auto* user_builder = get_builder(user);
      if (user_builder) {
        if (min_user_depth > user_builder->depth) {
          min_depth_user_builder = user_builder;
          min_user_depth = user_builder->depth;
        } else if (min_user_depth == user_builder->depth &&
                   user_builder->color != builder->color) {
          min_depth_user_builder = user_builder;
        }
      } else {
        all_users_assigned = false;
        break;
      }
    }

    if (!all_users_assigned) {
      continue;
    }

    VLOG(5) << "trying to reassign " << instruction->name()
            << " with min user depth " << min_user_depth;

    // we have found our match
    BuildContext* latest_builder{nullptr};
    if (min_depth_user_builder &&
        min_depth_user_builder->color == builder->color &&
        min_depth_user_builder != builder) {
      latest_builder = min_depth_user_builder;
    } else {
      // loop through all the builders of the same color and see if
      // we can move right
      for (auto* candidate : builders_for_color[builder->color]) {
        VLOG(5) << instruction->name() << " checking reassignment candidate "
                << candidate->name << ",depth=" << candidate->depth;
        if (candidate != builder && candidate->depth < min_user_depth &&
            candidate->depth >= builder->depth) {
          if (latest_builder && candidate->depth > latest_builder->depth) {
            latest_builder = candidate;
          } else {
            latest_builder = candidate;
          }
        }
      }
    }

    if (latest_builder) {
      VLOG(5) << "reassigning " << instruction->name() << " to later context "
              << latest_builder->name << ",color=" << latest_builder->color
              << ",depth=" << latest_builder->depth;
      builder_assignments[instruction] = latest_builder;
    }
  }

  for (auto* instruction : postorder) {
    if (IsMicrobatchLoop(instruction)) {
      // all of the inputs to the input tuple need to be marked as roots
      auto* input_tuple = instruction->operand(0);
      for (auto* operand : input_tuple->operands()) {
        auto* builder = get_builder(operand);
        if (builder && !builder->roots_added.contains(operand)) {
          builder->roots.push_back(operand);
          builder->roots_added.insert(operand);
        }
      }
      continue;
    }

    auto* builder = get_builder(instruction);
    if (builder) {
      TF_ASSIGN_OR_RETURN(
          auto* clone,
          AddCloneToContext(computation, builder, instruction, partition_,
                            builder_assignments, properties));
    }
  }

  // the traversal order of the instructions might cause contexts to get
  // added out of order
  // A -> context a, depth 0
  // B -> context b, depth 1
  // C -> context c, depth 0
  // D -> context b, depth 1
  std::stable_sort(
      builders.begin(), builders.end(),
      [](const auto& lhs, const auto& rhs) { return lhs->depth < rhs->depth; });

  for (auto&& builder : builders) {
    VLOG(5) << "building task computation " << builder->name << ":"
            << builder.get() << " color=" << builder->color
            << ",depth=" << builder->depth
            << ",call_to_recurse=" << builder->call_to_recurse;
    if (builder->call_to_recurse) {
      // this builder isn't actually used to break up into computations
      TF_RETURN_IF_ERROR(GroupComputationIntoTasks(
          builder->call_to_recurse, properties, builder->call_instruction));
      for (auto* user : builder->call_instruction->users()) {
        // no cloning, the original instruction will be used as inputs
        gte_root_clones[user] = user;
      }
      continue;
    }

    if (builder->roots.empty()) {
      // with optimization barriers, it can happen that a task was created
      // but it produces no useful output
      continue;
    }

    std::vector<HloInstruction*> roots;
    roots.reserve(builder->roots.size());
    for (auto* root : builder->roots) {
      roots.push_back(builder->clones[root]);
    }

    auto* root_tuple =
        builder->builder->AddInstruction(HloInstruction::CreateTuple(roots));

    bool is_backprop = false;
    TF_RETURN_IF_ERROR(builder->builder->ForEachInstruction(
        [&](const HloInstruction* instruction) {
          if (absl::StrContains(instruction->metadata().op_name(),
                                "transpose(jvp")) {
            is_backprop = true;
          }
          return absl::OkStatus();
        }));
    auto* task_comp = computation->parent()->AddComputationAndUnifyNamesAndIds(
        builder->builder->Build(root_tuple),
        /*is_entry=*/false);
    if (is_backprop && !absl::StartsWith(task_comp->name(), "bwd")) {
      std::string new_name = absl::StrCat("bwd.", task_comp->name());
      task_comp->SetAndSanitizeName(new_name);
    }

    std::vector<HloInstruction*> parameters;
    parameters.reserve(builder->parameters.size());
    for (auto* param : builder->parameters) {
      if (param->opcode() == HloOpcode::kParameter) {
        VLOG(5) << "task " << builder->name << " adding parameter "
                << param->name();
        parameters.push_back(param);
      } else {
        auto* clone = gte_root_clones[param];
        if (clone == nullptr) {
          return InvalidArgumentStrCat(
              param->name(), " in context ", builder->name, ":",
              absl::Hex(builder.get()),
              " has no intermediate root to use as a parameter");
        }
        VLOG(5) << "task " << builder->name << ":" << builder.get()
                << " adding cloned get-tuple-element " << clone->name()
                << " from param " << param->name() << " number "
                << parameters.size() << " " << param;
        parameters.push_back(clone);
      }
    }

    auto* call = computation->AddInstruction(
        HloInstruction::CreateCall(root_tuple->shape(), parameters, task_comp));
    AssignColor(call, builder->color);
    int64_t tuple_index = 0;
    for (auto* root : builder->roots) {
      if (gte_root_clones.contains(root)) {
        // this has already been cloned
        continue;
      }
      auto* gte_clone = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(call, tuple_index));
      properties.Clone(root, gte_clone);
      auto* matching_root = task_comp->root_instruction()->operand(tuple_index);
      AssignColor(gte_clone, builder->color);
      PropagateAxes(matching_root, gte_clone);
      if (matching_root->has_sharding()) {
        gte_clone->set_sharding(matching_root->sharding_ptr());
      }
      VLOG(5) << "cloned root " << root->name() << " to " << gte_clone->name()
              << " for color=" << builder->color;
      gte_root_clones[root] = gte_clone;
      TF_RETURN_IF_ERROR(root->ReplaceAllUsesWith(gte_clone));
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(root));
      ++tuple_index;
    }
  }

  TF_RETURN_IF_ERROR(RemoveUnusedInstructions(computation));

  for (auto* maybe_call : computation->instructions()) {
    if (maybe_call->opcode() == HloOpcode::kCall) {
      auto* comp = maybe_call->called_computations()[0];
      auto call_color = Color(maybe_call);
      if (!call_color.has_value()) {
        return InvalidArgumentStrCat(maybe_call->name(),
                                     " does not have a call");
      }
      for (auto* instruction : comp->MakeInstructionPostOrder()) {
        // this is a recorded resharding that we have to redo on the parameter
        if (instruction->IsCustomCall("Reshard")) {
          auto* operand = instruction->mutable_operand(0);
          VLOG(5) << comp->name() << " resharding parameter " << operand->name()
                  << " from " << instruction->name() << " to "
                  << instruction->sharding();
          operand->set_sharding(instruction->sharding_ptr());
          TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(operand));
          TF_RETURN_IF_ERROR(comp->RemoveInstruction(instruction));
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> MpmdComputationGrouper::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto properties = InstructionProperties::Create(module);
  TF_RETURN_IF_ERROR(GroupComputationIntoTasks(module->entry_computation(),
                                               properties, nullptr));

  // there may be "dangling" reshards in the entry computation if the argument
  // is no longer used in any tasks
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->IsCustomCall("Reshard")) {
      if (instruction->operand(0)->opcode() != HloOpcode::kParameter) {
        return InvalidArgumentStrCat(
            "orphaned reshard operation in entry computation does not have "
            "parameter operand");
      }
      TF_RETURN_IF_ERROR(
          instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
      TF_RETURN_IF_ERROR(
          module->entry_computation()->RemoveInstruction(instruction));
    }
  }
  return true;
}

}  // namespace xla
