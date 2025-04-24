/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_computation_fusion.h"

#include <algorithm>
#include <limits>
#include <memory>

#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/pjrt/legate/color_dfs.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla {

absl::Status MpmdComputationFusion::FuseComputations(
    HloComputation* parent, absl::Span<HloInstruction*> calls) {
  absl::flat_hash_set<HloInstruction*> produced_by_set;
  std::vector<HloInstruction*> parameters_needed;
  std::vector<std::pair<HloInstruction*, HloInstruction*>> roots_needed;
  absl::flat_hash_set<HloInstruction*> calls_included{calls.begin(),
                                                      calls.end()};
  absl::flat_hash_map<HloInstruction*, absl::InlinedVector<HloInstruction*, 2>>
      operands_added;

  VLOG(5) << "fusing " << calls.size() << " calls: " << calls.front()->name()
          << " ... " << calls.back()->name();
  for (HloInstruction* call : calls) {
    VLOG(5) << "  fuse " << call->name();
  }
  for (HloInstruction* call : calls) {
    HloComputation* computation = call->called_computations()[0];
    for (int64_t param = 0; param < call->operand_count(); ++param) {
      auto* operand = call->mutable_operand(param);
      if (!produced_by_set.contains(operand)) {
        if (!operands_added.contains(operand)) {
          VLOG(5) << "operand " << operand->name() << " of " << call->name()
                  << " is still an operand";
          parameters_needed.push_back(operand);
        }
        operands_added[operand].push_back(
            computation->parameter_instruction(param));
      }
    }

    for (auto* gte : call->users()) {
      produced_by_set.insert(gte);
      const bool is_root = [&] {
        if (gte->IsRoot()) {
          return true;
        }
        for (auto* gte_user : gte->users()) {
          if (!calls_included.contains(gte_user) || gte_user->IsRoot()) {
            return true;
          } else {
            VLOG(5) << "output " << gte->name() << " of " << call->name()
                    << " is used by " << gte_user->name();
          }
        }
        return false;
      }();
      if (is_root) {
        VLOG(5) << "output " << gte->name() << " of " << call->name()
                << " is still a root";
        roots_needed.push_back(
            {gte, computation->root_instruction()->mutable_operand(
                      gte->tuple_index())});
      }
    }
  }

  HloCloneContext context{parent->parent()};
  HloComputation::Builder builder{calls[0]->called_computations()[0]->name()};

  int64_t fused_param_number = 0;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> clone_map;
  absl::flat_hash_map<HloInstruction*, HloInstruction*> cloned_operand_params;
  for (auto* operand : parameters_needed) {
    auto* param = operands_added[operand].front();
    TF_ASSIGN_OR_RETURN(
        auto* new_param,
        builder.AddParameter(HloInstruction::CreateParameter(
            fused_param_number, param->shape(), param->name())));
    new_param->set_sharding(param->sharding_ptr());
    PropagateProperties(param, new_param);
    for (auto* param : operands_added[operand]) {
      clone_map[param] = new_param;
      VLOG(5) << operand->name() << " input maps to clone of " << param->name();
    }
    cloned_operand_params[operand] = new_param;
    ++fused_param_number;
  }

  for (auto* call : calls) {
    HloComputation* computation = call->called_computations()[0];
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kParameter) {
        clone_map[instruction] = cloned_operand_params[call->mutable_operand(
            instruction->parameter_number())];
      } else if (instruction != computation->root_instruction()) {
        absl::InlinedVector<HloInstruction*, 2> cloned_operands;
        for (auto* operand : instruction->operands()) {
          if (!clone_map.contains(operand)) {
            return InvalidArgumentStrCat(
                operand->name(), " was never added to clone map when fusing ");
          }
          cloned_operands.push_back(clone_map[operand]);
        }
        auto* new_instruction =
            builder.AddInstruction(instruction->CloneWithNewOperands(
                instruction->shape(), cloned_operands, &context));
        clone_map[instruction] = new_instruction;
      }
    }
    for (auto* user : call->users()) {
      auto* original_root =
          computation->root_instruction()->mutable_operand(user->tuple_index());
      auto* new_root = clone_map[original_root];
      cloned_operand_params[user] = new_root;
      VLOG(5) << user->name() << " output maps to clone of "
              << original_root->name();
    }
  }

  std::vector<HloInstruction*> fused_root_operands;
  for (auto [gte, root] : roots_needed) {
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

  std::vector<HloInstruction*> fused_call_operands;
  for (auto* operand : parameters_needed) {
    fused_call_operands.push_back(operand);
  }
  auto* new_call = parent->AddInstruction(HloInstruction::CreateCall(
      root_tuple->shape(), fused_call_operands, new_comp));

  PropagateColor(calls[0], new_call);

  // these need to be remapped to new get-tuple-element ops
  int64_t fused_output_tuple_index = 0;
  for (auto [gte, root] : roots_needed) {
    auto* new_gte =
        parent->AddInstruction(HloInstruction::CreateGetTupleElement(
            new_call, fused_output_tuple_index));
    new_gte->set_sharding(gte->sharding_ptr());
    PropagateProperties(gte, new_gte);
    VLOG(5) << "replacing old output " << gte->name() << " with new output "
            << new_gte->name();
    TF_RETURN_IF_ERROR(gte->ReplaceAllUsesWith(new_gte));
    if (gte->IsRoot()) {
      VLOG(5) << "replacing root " << gte->name() << " with new output "
              << new_gte->name();
      parent->set_root_instruction(new_gte);
    }
    ++fused_output_tuple_index;
  }

  for (auto iter = calls.rbegin(); iter != calls.rend(); ++iter) {
    auto* call = *iter;
    VLOG(5) << "removing call " << call->name();
    for (auto* user : call->users()) {
      if (!user->users().empty()) {
        return InternalStrCat(user->name(), " still has active user ",
                              user->users().front()->name());
      }
      if (user->IsRoot()) {
        return InternalStrCat(user->name(), " is still the root instruction");
      }
      VLOG(5) << "removing call user " << user->name();
      TF_RETURN_IF_ERROR(parent->RemoveInstruction(user));
    }
    auto* computation = call->called_computations()[0];
    TF_RETURN_IF_ERROR(parent->RemoveInstruction(call));
    TF_RETURN_IF_ERROR(
        parent->parent()->RemoveEmbeddedComputation(computation));
  }

  return absl::OkStatus();
}

bool MpmdComputationFusion::FusionMatch(const HloInstruction* lhs,
                                        const HloInstruction* rhs) {
  if (lhs->opcode() == HloOpcode::kWhile ||
      rhs->opcode() == HloOpcode::kWhile) {
    return false;
  }
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
  int64_t iter = 0;
  while (found_match) {
    VLOG(5) << "computating HLO ordering for fusion iteration";
    DependencyHloOrdering ordering{computation->parent()};

    struct FusionSet {
      absl::InlinedVector<HloInstruction*, 3> calls;
    };

    std::vector<std::unique_ptr<FusionSet>> sets;
    absl::flat_hash_map<HloInstruction*, FusionSet*> fusion_sets;

    auto postorder = ColorSortedPostorder(computation);

    std::vector<HloInstruction*> need_to_visit;
    absl::flat_hash_map<HloInstruction*, int64_t> postorder_index;
    for (auto* instruction : postorder) {
      if (instruction->opcode() == HloOpcode::kCall ||
          instruction->opcode() == HloOpcode::kWhile) {
        need_to_visit.push_back(instruction);
        postorder_index[instruction] = postorder_index.size();
      }
    }
    sets.reserve(need_to_visit.size());

    if (VLOG_IS_ON(5)) {
      int64_t num_calls = 0;
      for (auto* instruction : postorder) {
        if (instruction->opcode() == HloOpcode::kCall) {
          ++num_calls;
          VLOG(5) << computation->name() << " has call " << instruction->name()
                  << " with color " << Color(instruction).value_or("none");
        }
      }
      VLOG(5) << computation->name() << " has a total of " << num_calls
              << " calls starting fusion iteration";
    }

    // first try to move instructions forward
    std::vector<std::pair<HloInstruction*, HloInstruction*>> fusion_pairs;
    for (int l = 0; l < need_to_visit.size(); ++l) {
      HloInstruction* lhs = need_to_visit[l];
      if (lhs->opcode() == HloOpcode::kWhile) {
        continue;
      }

      absl::InlinedVector<HloInstruction*, 1> self = {lhs};

      auto current_fusion_set =
          [&]() -> absl::Span<const HloInstruction* const> {
        if (fusion_sets.contains(lhs)) {
          // the front of the fusion set determines safety of reordering
          // due to the DFS use
          return fusion_sets[lhs]->calls;
        }
        return self;
      }();

      VLOG(5) << "trying to find fusions of " << lhs->name();
      HloInstruction* match{nullptr};
      for (int r = l + 1; r < need_to_visit.size(); ++r) {
        HloInstruction* rhs = need_to_visit[r];
        if (FusionMatch(lhs, rhs)) {
          match = rhs;
          break;
        }
        const bool can_reorder = [&]() {
          for (auto iter = current_fusion_set.rbegin();
               iter != current_fusion_set.rend(); ++iter) {
            if (ordering.ExecutesBefore(*iter, rhs)) {
              VLOG(5) << "" << (*iter)->name() << " executes before "
                      << rhs->name() << ", stopping search for fusion of "
                      << lhs->name();
              return false;
            };
          }
          return true;
        }();

        if (!can_reorder) {
          break;
        }
      }
      if (match) {
        VLOG(5) << lhs->name() << " matches " << match->name()
                << " for forward fusion";
        if (fusion_sets.contains(match)) {
          return InvalidArgumentStrCat(match->name(),
                                       " was already assigned a fusion parner");
        }
        auto& set = fusion_sets[lhs];
        if (set == nullptr) {
          set = sets.emplace_back(std::make_unique<FusionSet>()).get();
          set->calls.push_back(lhs);
        }
        set->calls.push_back(match);
        fusion_sets[match] = set;
      }
    }

    // now try to move instructions backward
    for (int r = 0; r < need_to_visit.size(); ++r) {
      HloInstruction* rhs = need_to_visit[r];
      if (rhs->opcode() == HloOpcode::kWhile) {
        continue;
      }

      absl::InlinedVector<HloInstruction*, 1> self = {rhs};

      auto current_fusion_set =
          [&]() -> absl::Span<const HloInstruction* const> {
        if (fusion_sets.contains(rhs)) {
          // the front of the fusion set determines safety of reordering
          // due to the DFS use
          return fusion_sets[rhs]->calls;
        }
        return self;
      }();

      VLOG(5) << "trying to find fusions of " << rhs->name();

      HloInstruction* match{nullptr};
      for (int l = r - 1; l >= 0; --l) {
        HloInstruction* lhs = need_to_visit[l];
        if (FusionMatch(lhs, rhs)) {
          match = lhs;
          break;
        }
        const bool can_reorder = [&]() {
          for (auto iter = current_fusion_set.rbegin();
               iter != current_fusion_set.rend(); ++iter) {
            if (ordering.ExecutesBefore(lhs, *iter)) {
              VLOG(5) << lhs->name() << " executes before " << (*iter)->name()
                      << ", stopping search for fusion of " << rhs->name();
              return false;
            };
          }
          return true;
        }();

        if (!can_reorder) {
          break;
        }
      }
      if (match) {
        VLOG(5) << match->name() << " matches " << rhs->name()
                << " for backward fusion";
        auto& set = fusion_sets[match];
        if (set == nullptr) {
          set = sets.emplace_back(std::make_unique<FusionSet>()).get();
          set->calls.push_back(match);
        }
        if (!fusion_sets.contains(rhs)) {
          set->calls.push_back(rhs);
          fusion_sets[rhs] = set;
        }
      }
    }

    found_match = !fusion_sets.empty();
    changed |= found_match;

    for (auto& set : sets) {
      // the fusion requires that these come in DFS order so that producers
      // are guaranteed to be visited before consumers
      std::sort(set->calls.begin(), set->calls.end(),
                [&](HloInstruction* lhs, HloInstruction* rhs) {
                  return postorder_index[lhs] < postorder_index[rhs];
                });
      TF_RETURN_IF_ERROR(FuseComputations(
          computation, {set->calls.data(), set->calls.size()}));
    }
    ++iter;
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

  if (changed) {
    HloDCE dce{};
    TF_ASSIGN_OR_RETURN(bool deleted, dce.Run(module));
  }

  return changed;
}

}  // namespace xla
