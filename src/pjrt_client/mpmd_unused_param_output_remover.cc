#include "xla/pjrt/legate/mpmd_unused_param_output_remover.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_unused_param_output_remover.h"

namespace xla {

absl::StatusOr<bool> MpmdUnusedParamOutputRemover::Visit(
    HloInstruction* call_instruction) {
  VLOG(5) << "checking " << call_instruction->name()
          << " for unused parameters/outputs";
  auto* module = call_instruction->parent()->parent();
  auto* called_computation = call_instruction->called_computations()[0];

  absl::flat_hash_set<HloInstruction*> deleted_parameters;
  auto color = Color(call_instruction);
  CHECK(color.has_value());
  bool output_removed = false;
  for (auto* user : call_instruction->users()) {
    auto* matching_root =
        called_computation->root_instruction()->operand(user->tuple_index());
    if (user->users().empty() ||
        matching_root->opcode() == HloOpcode::kParameter) {
      output_removed = true;
      break;
    }
  }

  for (auto* param : called_computation->parameter_instructions()) {
    if (param->users().empty() ||
        (param->users().size() == 1 &&
         param->users().front() == called_computation->root_instruction())) {
      deleted_parameters.insert(param);
    }
  }

  if (!output_removed && deleted_parameters.empty()) {
    return false;
  }

  HloCloneContext context{call_instruction->parent()->parent()};
  HloComputation::Builder builder{called_computation->name()};

  absl::flat_hash_map<const HloInstruction*, HloInstruction*> clone_map;

  int64_t parameter_number = 0;
  std::vector<HloInstruction*> new_parameters;
  for (auto* instruction : called_computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      if (!deleted_parameters.contains(instruction)) {
        new_parameters.push_back(
            call_instruction->mutable_operand(instruction->parameter_number()));
        TF_ASSIGN_OR_RETURN(
            auto* new_param,
            builder.AddParameter(HloInstruction::CreateParameter(
                parameter_number++, instruction->shape(),
                instruction->name())));
        if (instruction->has_sharding()) {
          new_param->set_sharding(instruction->sharding_ptr());
        }
        clone_map[instruction] = new_param;
        AssignColor(new_param, *color);
      }
    } else if (instruction != called_computation->root_instruction()) {
      absl::InlinedVector<HloInstruction*, 2> new_operands;
      for (auto* operand : instruction->operands()) {
        auto iter = clone_map.find(operand);
        if (iter == clone_map.end()) {
          return InvalidArgumentStrCat(
              instruction->name(),
              " missing in clone map for MpmdUnusedParamOutputRemover");
        }
        new_operands.push_back(iter->second);
      }
      auto* clone = builder.AddInstruction(instruction->CloneWithNewOperands(
          instruction->shape(), new_operands, &context));
      AssignColor(clone, *color);
      clone_map[instruction] = clone;
    }
  }

  std::vector<HloInstruction*> new_root_operands;
  std::vector<HloInstruction*> kept_outputs;
  for (auto* user : call_instruction->users()) {
    auto* matching_root =
        called_computation->root_instruction()->operand(user->tuple_index());
    if (matching_root->opcode() == HloOpcode::kParameter) {
      auto* alias_gte =
          call_instruction->mutable_operand(matching_root->parameter_number());
      TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(alias_gte));
      VLOG(5) << "removing alias root/param " << user->name() << " of "
              << call_instruction->name() << " with " << alias_gte->name();
      TF_RETURN_IF_ERROR(call_instruction->parent()->RemoveInstruction(user));
    } else if (user->users().empty() &&
               user != call_instruction->parent()->root_instruction()) {
      VLOG(5) << "removing unused output " << user->name() << " of "
              << call_instruction->name();
      TF_RETURN_IF_ERROR(call_instruction->parent()->RemoveInstruction(user));
    } else {
      auto iter = clone_map.find(matching_root);
      if (iter == clone_map.end()) {
        return InvalidArgumentStrCat(
            matching_root->name(),
            " missing in clone map for MpmdUnusedParamOutputRemover");
      }
      VLOG(5) << "keeping output " << user->name() << " of "
              << call_instruction->name();
      new_root_operands.push_back(iter->second);
      kept_outputs.push_back(user);
    }
  }

  auto* new_root_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple(new_root_operands));
  auto* new_computation =
      call_instruction->parent()->parent()->AddComputationAndUnifyNamesAndIds(
          builder.Build(new_root_tuple), /*is_entry=*/false);

  auto* new_call =
      call_instruction->parent()->AddInstruction(HloInstruction::CreateCall(
          new_root_tuple->shape(), new_parameters, new_computation));
  AssignColor(new_call, *color);

  for (int64_t gte_index = 0; gte_index < kept_outputs.size(); ++gte_index) {
    auto* user = kept_outputs[gte_index];
    auto* new_gte = call_instruction->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(new_call, gte_index));
    VLOG(5) << "replacing output " << user->name() << " of "
            << call_instruction->name() << " with clone " << new_gte->name();
    if (user->has_sharding()) {
      new_gte->set_sharding(user->sharding_ptr());
    }
    PropagateColor(user, new_gte);
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(new_gte));
    TF_RETURN_IF_ERROR(call_instruction->parent()->RemoveInstruction(user));
  }

  TF_RETURN_IF_ERROR(
      call_instruction->parent()->RemoveInstruction(call_instruction));
  TF_RETURN_IF_ERROR(called_computation->parent()->RemoveEmbeddedComputation(
      called_computation));

  return true;
}

absl::StatusOr<bool> MpmdUnusedParamOutputRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  std::vector<HloComputation*> to_visit = {module->entry_computation()};

  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    auto postorder = computation->MakeInstructionPostOrder();
    // reverse iterator
    for (auto iter = postorder.rbegin(); iter != postorder.rend(); ++iter) {
      HloInstruction* instruction = *iter;
      if (instruction->opcode() == HloOpcode::kWhile) {
        to_visit.push_back(instruction->called_computations()[0]);
      } else if (instruction->opcode() == HloOpcode::kCall) {
        TF_ASSIGN_OR_RETURN(bool comp_changed, Visit(instruction));
        changed |= comp_changed;
      }
    }
  }
  return changed;
}

}  // namespace xla