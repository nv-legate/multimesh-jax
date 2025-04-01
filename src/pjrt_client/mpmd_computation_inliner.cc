#include "xla/pjrt/legate/mpmd_computation_inliner.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/service/call_inliner.h"
#include "xla/service/tuple_simplifier.h"

namespace xla {

absl::StatusOr<bool> MpmdComputationInliner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // recolor everything to have a unique color per-task

  std::vector<HloComputation*> to_visit = {module->entry_computation()};
  while (!to_visit.empty()) {
    auto* computation = to_visit.back();
    to_visit.pop_back();

    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kCall) {
        auto* comp = instruction->called_computations()[0];
        std::optional<std::string> color = Color(instruction);
        if (!color.has_value()) {
          return InvalidArgumentStrCat(module->name(), " has call ",
                                       instruction->name(),
                                       " without color assigned");
        }
        VLOG(5) << "visiting call " << instruction->name() << " with color "
                << *color << " while inlining";

        // after inlining, all of the parameter shardings will be erased
        // we have to record them now so they don't get lost
        for (auto* param : comp->parameter_instructions()) {
          if (param->users().empty()) {
            RemoveColor(param);
            param->set_frontend_attributes({});
          } else if (param->has_sharding()) {
            auto* reshard =
                comp->AddInstruction(HloInstruction::CreateCustomCall(
                    param->shape(), {param}, "Reshard"));
            reshard->set_sharding(param->sharding_ptr());
            AssignColor(reshard, *color);
            TF_RETURN_IF_ERROR(param->ReplaceAllUsesWith(reshard));
          }
        }
      } else if (instruction->opcode() == HloOpcode::kWhile) {
        to_visit.push_back(instruction->called_computations()[0]);
      }
    }
  }

  CallInliner inliner;
  TF_ASSIGN_OR_RETURN(bool changed, inliner.Run(module));

  if (!changed) {
    return false;
  }

  TupleSimplifier simplifier;
  TF_ASSIGN_OR_RETURN(changed, simplifier.Run(module));

  absl::flat_hash_set<HloInstruction*> roots;
  auto* root = module->entry_computation()->root_instruction();
  if (root->shape().IsTuple()) {
    roots.insert(root->operands().begin(), root->operands().end());
  } else {
    roots.insert(root);
  }

  return true;
}

}  // namespace xla