/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_computation_grouper.h"

#include <algorithm>
#include <limits>
#include <optional>

#include "mpmd_instruction.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/color_dfs.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/util.h"

namespace xla {

namespace {

bool IsRoot(HloInstruction* instruction) {
  auto* root = instruction->parent()->root_instruction();
  if (root->opcode() == HloOpcode::kTuple) {
    return absl::c_any_of(instruction->users(),
                          [&](HloInstruction* user) { return user == root; });
  }
  return instruction == root;
}

bool IsAliasedGetTupleElement(const HloInstruction* instruction) {
  switch (instruction->operand(0)->opcode()) {
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kTuple:
      return true;
    default:
      return false;
  }
}

bool IsMicrobatchLoop(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kWhile &&
         instruction->has_backend_config();
}

// Returns whether the instruction should never be included directly in a task
// and should instead always be passed as an inter-task operand
bool IsAlwaysTaskOperand(HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kParameter:
      return true;
    case HloOpcode::kGetTupleElement:
      return IsMicrobatchLoop(instruction->operand(0)) ||
             instruction->operand(0)->opcode() == HloOpcode::kParameter;
    case HloOpcode::kCustomCall:
      return instruction->IsCustomCall(kCustomCallSliceOffset) ||
             instruction->IsCustomCall(kCustomCallDummyOperation);
    default:
      return false;
  }
}

// Returns whether the instruction should always be recomputed in every task
// instead of being passed as an operand between tasks.  This includes some
// no-ops like constants that should never be allocated inter-task buffers or
// operations like broadcast that should be recomputed to limit memory
// allocations
bool ReplicateInstruction(const HloInstruction* instruction,
                          const InstructionProperties& properties) {
  if (properties.DerivedInput(instruction) || !IsAssignedColor(instruction) ||
      instruction->IsCustomCall(kCustomCallArgumentRecolor)) {
    return true;
  }

  switch (instruction->opcode()) {
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kTuple:
      return true;
    case HloOpcode::kGetTupleElement:
      return IsAliasedGetTupleElement(instruction);
    default:
      return false;
  }
}

bool IncludeDirectlyInBuilder(HloInstruction* instruction,
                              const InstructionProperties& properties) {
  return !ReplicateInstruction(instruction, properties) &&
         !IsAlwaysTaskOperand(instruction);
}

bool IncludeIndirectlyInBuilder(HloInstruction* instruction,
                                const InstructionProperties& properties) {
  return ReplicateInstruction(instruction, properties) &&
         !IsAlwaysTaskOperand(instruction);
}

absl::Status ApplyInterTaskReshards(HloInstruction* call) {
  auto* comp = call->called_computations()[0];
  auto call_color = Color(call);
  if (!call_color.has_value()) {
    return InvalidArgumentStrCat(call->name(), " does not have a color");
  }
  for (auto* instruction : comp->MakeInstructionPostOrder()) {
    // this is a recorded resharding that we have to redo on the parameter
    if (instruction->IsCustomCall(kCustomCallArgumentRecolor)) {
      auto* operand = instruction->mutable_operand(0);
      VLOG(5) << comp->name() << " resharding parameter " << operand->name()
              << " from " << instruction->name() << " to "
              << instruction->sharding();
      operand->set_sharding(instruction->sharding_ptr());
      TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(operand));
      TF_RETURN_IF_ERROR(comp->RemoveInstruction(instruction));
    }
  }
  return absl::OkStatus();
}

class BuildContext {
 public:
  explicit BuildContext(std::string name,
                        std::optional<std::string> color = std::nullopt)
      : builder_(HloComputation::Builder{name}),
        name_(name),
        color_(color.value_or("none")),
        call_instruction_(nullptr) {}

  explicit BuildContext(HloInstruction* parent) : call_instruction_(parent) {}

  bool IsParentCallInstruction() const { return call_instruction_; }

  absl::Status ReplaceRootsWithCallOutputs(
      HloComputation* computation,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>& clone_map) {
    for (auto* root : roots_) {
      auto* gte_clone = clone_map[root];
      VLOG(5) << name_ << " replacing root " << root->name() << " with "
              << gte_clone->name();
      TF_RETURN_IF_ERROR(root->ReplaceAllUsesWith(gte_clone));
      if (root == computation->root_instruction()) {
        computation->set_root_instruction(gte_clone);
      }
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(root));
    }
    return absl::OkStatus();
  }

  void AddConsumer(HloInstruction* root, BuildContext* context) {
    if (producers_added_.contains(this)) {
      LOG(FATAL) << "operand " << root->name()
                 << " produces circular dependency";
    }

    AddRoot(root);

    if (!consumers_added_.contains(context)) {
      consumers_.push_back(context);
    }
    if (!context->producers_added_.contains(this)) {
      context->producers_.push_back(this);
    }
  }

  const std::vector<BuildContext*>& Producers() const { return producers_; }

  const std::vector<BuildContext*>& Consumers() const { return consumers_; }

  absl::Status ReplaceWithCall(
      HloComputation* computation, InstructionProperties& properties,
      absl::flat_hash_map<HloInstruction*, HloInstruction*>& clone_map);

  void AddReplicatedOperandDfs(HloInstruction* instruction,
                               const InstructionProperties& properties) {
    absl::InlinedVector<HloInstruction*, 6> to_visit;

    auto include_in_dfs = [&](HloInstruction* instruction) {
      return !IsAssignedColor(instruction) && !IsAlwaysTaskOperand(instruction);
    };

    for (auto* operand : instruction->mutable_operands()) {
      if (!instructions_added_.contains(operand)) {
        if (IncludeIndirectlyInBuilder(operand, properties)) {
          to_visit.push_back(operand);
        }
      }
    }

    if (to_visit.empty()) {
      return;
    }

    absl::flat_hash_map<HloInstruction*, bool> visited;
    while (!to_visit.empty()) {
      HloInstruction* next = to_visit.back();
      if (!visited.contains(next)) {
        for (auto* operand : next->mutable_operands()) {
          if (IncludeIndirectlyInBuilder(operand, properties) &&
              !instructions_added_.contains(operand)) {
            to_visit.push_back(operand);
          }
        }
        visited[next] = false;
        continue;
      }

      to_visit.pop_back();
      if (visited[next]) {
        continue;
      }

      VLOG(5) << "uncolored operand " << next->name() << " added to builder "
              << name_ << ":" << this;
      instruction_dfs_.push_back(next);
      instructions_added_.insert(next);
      visited[next] = true;
    }
  }

  void AddRoot(HloInstruction* root) {
    if (root->shape().IsTuple()) {
      LOG(FATAL) << "Tuple root " << root->name() << " not supported yet";
    }
    if (!roots_added_.contains(root)) {
      VLOG(5) << "operand " << root->name() << " added as root for builder "
              << name_ << ":" << this;
      roots_.push_back(root);
      roots_added_.insert(root);
    }
  }

  void AddInstruction(HloInstruction* instruction,
                      const InstructionProperties& properties) {
    if (!instructions_added_.contains(instruction)) {
      AddReplicatedOperandDfs(instruction, properties);
      VLOG(5) << "instruction " << instruction->name() << " added to builder "
              << name_ << ":" << this;
      instruction_dfs_.push_back(instruction);
      instructions_added_.insert(instruction);
    }
  }

  void Print(std::ostream& os) const { os << name_ << ":" << this; }

  absl::Status Build(
      const absl::flat_hash_map<HloInstruction*, BuildContext*>& builders);

  HloInstruction* CallInstruction() const { return call_instruction_; }

 private:
  std::string name_;
  std::string color_{"none"};
  absl::flat_hash_map<HloInstruction*, HloInstruction*> clones_;
  std::optional<HloComputation::Builder> builder_;
  std::vector<HloInstruction*> parameters_;
  std::vector<HloInstruction*> roots_;
  std::vector<BuildContext*> producers_;
  std::vector<BuildContext*> consumers_;
  std::vector<HloInstruction*> instruction_dfs_;
  absl::flat_hash_set<BuildContext*> producers_added_;
  absl::flat_hash_set<BuildContext*> consumers_added_;
  absl::flat_hash_set<HloInstruction*> roots_added_;
  absl::flat_hash_set<HloInstruction*> instructions_added_;
  HloInstruction* call_instruction_{nullptr};
};

std::ostream& operator<<(std::ostream& os, const BuildContext& context) {
  context.Print(os);
  return os;
}

absl::Status BuildContext::ReplaceWithCall(
    HloComputation* computation, InstructionProperties& properties,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& clone_map) {
  if (roots_.empty()) {
    return absl::OkStatus();
  }

  VLOG(5) << "finalizing task computation " << *this;
  std::vector<HloInstruction*> roots;
  roots.reserve(roots_.size());
  for (auto* root : roots_) {
    auto iter = clones_.find(root);
    if (iter == clones_.end()) {
      return InvalidArgumentStrCat("root ", root->name(), " for task ", name_,
                                   " has not been cloned");
    }
    roots.push_back(iter->second);
  }

  auto* root_tuple =
      builder_->AddInstruction(HloInstruction::CreateTuple(roots));

  auto* task_computation =
      computation->parent()->AddComputationAndUnifyNamesAndIds(
          builder_->Build(root_tuple),
          /*is_entry=*/false);

  std::vector<HloInstruction*> parameters;
  parameters.reserve(parameters_.size());
  for (auto* param : parameters_) {
    if (param->opcode() == HloOpcode::kParameter) {
      parameters.push_back(param);
    } else {
      auto* clone = clone_map[param];
      if (clone == nullptr) {
        return InvalidArgumentStrCat(
            param->name(), " in context ", name_, ":", absl::Hex(this),
            " has no intermediate root to use as a parameter");
      }
      parameters.push_back(clone);
    }
  }

  auto* call = computation->AddInstruction(HloInstruction::CreateCall(
      root_tuple->shape(), parameters, task_computation));
  AssignColor(call, color_);
  int64_t tuple_index = 0;
  for (auto* root : roots_) {
    if (clone_map.contains(root)) {
      // this has already been cloned
      ++tuple_index;
      continue;
    }
    auto* gte_clone = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(call, tuple_index));
    properties.Clone(root, gte_clone);
    auto* matching_root =
        task_computation->root_instruction()->operand(tuple_index);
    AssignColor(gte_clone, color_);
    PropagateAxes(matching_root, gte_clone);
    if (matching_root->has_sharding()) {
      gte_clone->set_sharding(matching_root->sharding_ptr());
    }
    clone_map[root] = gte_clone;
    ++tuple_index;
  }
  return absl::OkStatus();
}

absl::Status BuildContext::Build(
    const absl::flat_hash_map<HloInstruction*, BuildContext*>& builders) {
  VLOG(5) << "building context " << *this;

  absl::flat_hash_set<HloInstruction*> parameters_added;
  for (auto* instruction : instruction_dfs_) {
    if (IsRoot(instruction)) {
      AddRoot(instruction);
    }
    absl::InlinedVector<HloInstruction*, 6> new_operands;
    VLOG(5) << "adding " << instruction->name() << " to " << *this;
    for (auto* operand : instruction->mutable_operands()) {
      auto iter = clones_.find(operand);
      if (iter == clones_.end()) {
        TF_ASSIGN_OR_RETURN(
            auto* clone,
            builder_->AddParameter(HloInstruction::CreateParameter(
                parameters_.size(), operand->shape(), operand->name())));
        if (operand->has_sharding()) {
          // pass on replication to the cloned operand
          clone->set_sharding(operand->sharding_ptr());
        }

        auto iter = builders.find(operand);
        if (iter != builders.end()) {
          // this is produced by a previous task rather than being
          // a parameter or fixed task operand
          iter->second->AddConsumer(operand, this);
        }

        AssignColor(clone, color_);
        PropagateAxes(operand, clone);

        VLOG(5) << instruction->name() << " cloned operand " << operand->name()
                << " as parameter number " << parameters_.size() << " on "
                << *this;

        parameters_.push_back(operand);
        clones_[operand] = clone;
        new_operands.push_back(clone);
      } else {
        auto* clone = iter->second;
        new_operands.push_back(clone);
      }
    }
    auto* clone = builder_->AddInstruction(
        instruction->CloneWithNewOperands(instruction->shape(), new_operands));
    PropagateProperties(instruction, clone);
    if (instruction->has_sharding()) {
      clone->set_sharding(instruction->sharding_ptr());
    }
    VLOG(5) << "cloned " << instruction->name() << " in context " << *this;
    clones_[instruction] = clone;
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status MpmdComputationGrouper::GroupComputationIntoTasks(
    HloComputation* computation, InstructionProperties& properties,
    HloInstruction* parent_call) {
  std::vector<std::unique_ptr<BuildContext>> builders;
  absl::flat_hash_map<HloInstruction*, BuildContext*> builder_assignments;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> gte_root_clones;

  // first scrub the color from all instructions that should be replicated
  for (auto* instruction : computation->instructions()) {
    if (IsAlwaysTaskOperand(instruction)) {
      gte_root_clones[instruction] = instruction;
    }
  }

  auto postorder = ColorSortedPostorder(computation);

  std::optional<std::string> last_color{std::nullopt};
  BuildContext* current_builder{nullptr};
  for (auto* instruction : postorder) {
    VLOG(5) << "visiting " << instruction->name() << " in computation grouper";
    if (instruction->IsCustomCall(kCustomCallRootTupleRecolor)) {
      // these should never be grouped into a computation
      // but their operand will be a root output of their task
      HloInstruction* operand = instruction->mutable_operand(0);
      builder_assignments[operand]->AddRoot(operand);
      continue;
    }
    if (IsMicrobatchLoop(instruction)) {
      // this is a MultiMesh while loop
      builders.push_back(std::make_unique<BuildContext>(instruction));
      // all inputs to the loop should be roots of a precursor task
      for (auto* operand :
           instruction->mutable_operand(0)->mutable_operands()) {
        auto iter = builder_assignments.find(operand);
        if (iter != builder_assignments.end()) {
          iter->second->AddRoot(operand);
        }
      }
      for (auto* user : instruction->users()) {
        gte_root_clones[user] = user;
      }
      last_color = std::nullopt;
      continue;
    }

    if (IncludeDirectlyInBuilder(instruction, properties)) {
      auto color = Color(instruction);
      if (!color.has_value()) {
        return InvalidArgumentStrCat("instruction ", instruction->name(),
                                     " has no color");
      }
      if (last_color != color) {
        std::string name = *color;
        if (parent_call && parent_call->opcode() == HloOpcode::kWhile) {
          absl::StrAppend(&name, "_loop");
        }
        builders.push_back(std::make_unique<BuildContext>(name, color));
        current_builder = builders.back().get();
      }
      current_builder->AddInstruction(instruction, properties);
      builder_assignments[instruction] = current_builder;
      last_color = color;
    }
  }

  for (auto&& builder : builders) {
    if (builder->CallInstruction()) {
      // this builder isn't actually used to break up into computations
      TF_RETURN_IF_ERROR(GroupComputationIntoTasks(
          builder->CallInstruction()->called_computations()[0], properties,
          builder->CallInstruction()));
      for (auto* user : builder->CallInstruction()->users()) {
        // no cloning, the original instruction will be used as inputs
        gte_root_clones[user] = user;
      }
      continue;
    } else {
      TF_RETURN_IF_ERROR(builder->Build(builder_assignments));
    }
  }

  // Create the task computations and the corresponding get-tuple-element
  // outputs
  for (auto&& builder : builders) {
    if (!builder->IsParentCallInstruction()) {
      TF_RETURN_IF_ERROR(
          builder->ReplaceWithCall(computation, properties, gte_root_clones));
    }
  }

  // Replace all uses of the original instructions with their new task outputs
  for (auto&& builder : builders) {
    TF_RETURN_IF_ERROR(
        builder->ReplaceRootsWithCallOutputs(computation, gte_root_clones));
  }

  TF_RETURN_IF_ERROR(RemoveUnusedInstructions(computation));

  for (auto* maybe_call : computation->instructions()) {
    VLOG(5) << "have " << maybe_call->ToString() << " in final module";
    if (maybe_call->opcode() == HloOpcode::kCall) {
      for (auto* instruction :
           maybe_call->called_computations()[0]->instructions()) {
        VLOG(5) << "   " << maybe_call->name() << " has instruction "
                << instruction->ToString();
      }
      TF_RETURN_IF_ERROR(ApplyInterTaskReshards(maybe_call));
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
    if (instruction->IsCustomCall(kCustomCallArgumentRecolor)) {
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
