/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_microbatch_loop_canonicalizer.h"

#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/service/call_inliner.h"
#include "xla/hlo/transforms/simplifiers//tuple_simplifier.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

absl::StatusOr<bool> MpmdMicrobatchLoopCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // ensure that calls are inlined
  CallInliner inliner{};
  TF_ASSIGN_OR_RETURN(bool inlined, inliner.Run(module));

  TupleSimplifier simplifier{};
  TF_ASSIGN_OR_RETURN(bool simplified, simplifier.Run(module));

  bool changed = inlined || simplified;

  absl::flat_hash_set<HloInstruction*> microbatch_inputs;
  absl::flat_hash_set<HloInstruction*> microbatch_accumulators;
  absl::flat_hash_set<int64_t> loop_accumulator_operand_indices;

  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    VLOG(3) << "visiting instruction " << instruction->name();
    if (instruction->IsCustomCall("MicrobatchInit")) {
      VLOG(3) << "found microbatch init " << instruction->name();

      // follow this to a loop
      absl::InlinedVector<HloInstruction*, 6> to_visit{instruction};

      if (instruction->users().size() != 1) {
        return InvalidArgumentStrCat(instruction->name(),
                                     " should have a single user");
      }

      auto continue_user_tree = [](HloInstruction* instr) {
        switch (instr->opcode()) {
          case HloOpcode::kReshape:
          case HloOpcode::kBroadcast:
          case HloOpcode::kCustomCall:
            return true;
          default:
            return false;
        }
      };

      HloInstruction* loop_input = instruction->users().front();
      HloInstruction* loop_input_operand = instruction;
      while (continue_user_tree(loop_input)) {
        if (loop_input->users().size() != 1) {
          return InvalidArgumentStrCat(instruction->name(),
                                       " has multiple users in the input tree");
        }
        loop_input_operand = loop_input;
        loop_input = loop_input->users().front();
      }

      auto* copy = module->entry_computation()->AddInstruction(
          HloInstruction::CreateUnary(loop_input_operand->shape(),
                                      HloOpcode::kCopy,
                                      loop_input_operand->mutable_operand(0)));
      microbatch_inputs.insert(copy);
      copy->set_raw_backend_config_string(
          instruction->raw_backend_config_string());
      PropagateProperties(instruction, copy);
      TF_RETURN_IF_ERROR(loop_input_operand->ReplaceAllUsesWith(copy));
      TF_RETURN_IF_ERROR(
          instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
      TF_RETURN_IF_ERROR(
          module->entry_computation()->RemoveInstruction(instruction));

      changed = true;
    } else if (instruction->IsCustomCall("Microbatch")) {
      microbatch_inputs.insert(instruction);
    } else if (instruction->opcode() == HloOpcode::kWhile) {
      auto* body = instruction->called_computations()[0];
      auto* body_root = body->root_instruction();
      auto* input_tuple = instruction->mutable_operand(0);
      for (int64_t index = 0; index < input_tuple->operand_count(); ++index) {
        auto* input = input_tuple->operand(index);
        if (microbatch_inputs.contains(input)) {
          changed = true;
          instruction->set_raw_backend_config_string(
              input->raw_backend_config_string());
          break;
        }
      }

      // this is a custom while loop
      if (instruction->has_backend_config()) {
        // check for constant inputs to the loop, these will be loop
        // accumulators so we make copies to manage them correctly
        for (int64_t index = 0; index < input_tuple->operand_count(); ++index) {
          auto* operand = input_tuple->mutable_operand(index);
          if (operand->opcode() == HloOpcode::kConstant) {
            auto* copy = module->entry_computation()->AddInstruction(
                HloInstruction::CreateUnary(operand->shape(), HloOpcode::kCopy,
                                            operand));
            TF_RETURN_IF_ERROR(input_tuple->ReplaceOperandWith(index, copy));
            if (operand->users().empty()) {
              TF_RETURN_IF_ERROR(
                  module->entry_computation()->RemoveInstruction(operand));
            }
          }
        }
      }
    }
  }

  TF_ASSIGN_OR_RETURN(bool slices_changed,
                      CanonicalizeSlices(module->entry_computation(), nullptr));

  return changed || slices_changed;
}

absl::StatusOr<bool> MpmdMicrobatchLoopCanonicalizer::CanonicalizeSlices(
    HloComputation* computation, HloInstruction* parent_loop) {
  bool changed = false;
  HloInstruction* loop_slice = nullptr;
  for (auto* instruction : computation->instructions()) {
    if (instruction->IsCustomCall("Microbatch")) {
      changed = true;

      if (instruction->users().size() != 1) {
        return InvalidArgumentStrCat(instruction->name(),
                                     " should have a single microbatch user");
      }
      TF_RETURN_IF_ERROR(
          instruction->ReplaceAllUsesWith(instruction->mutable_operand(0)));
      TF_RETURN_IF_ERROR(
          computation->parent()->entry_computation()->RemoveInstruction(
              instruction));
    } else if (instruction->IsCustomCall("MicrobatchSlice")) {
      auto* ds = instruction->mutable_operand(0);
      auto* operand_to_replace = ds;
      if (ds->IsCustomCall("Sharding") || ds->IsCustomCall("AutoSharding")) {
        ds = ds->mutable_operand(0);
      }

      if (ds->opcode() != HloOpcode::kDynamicSlice) {
        return InvalidArgumentStrCat(
            "Microbatch slice should have a single dynamic-slice argument, "
            "found ",
            ds->name());
      }

      CHECK(parent_loop != nullptr);
      TF_ASSIGN_OR_RETURN(
          auto loop_config,
          GetMicrobatchConfig(std::string(instruction->name()),
                              parent_loop->raw_backend_config_string()));
      const int64_t slice_index = loop_config.slice_dim;
      auto* slice_offset_operand = ds->mutable_operand(slice_index + 1);
      if (loop_slice == nullptr) {
        VLOG(5) << "found new microbatch slice offset "
                << slice_offset_operand->name();
        auto* copy =
            computation->AddInstruction(HloInstruction::CreateCustomCall(
                slice_offset_operand->shape(), {}, kCustomCallSliceOffset));
        copy->SetAndSanitizeName("slice-offset");
        loop_slice = copy;
      }

      VLOG(5) << "using microbatch slice offset " << loop_slice->name()
              << " instead of " << slice_offset_operand->name();
      TF_RETURN_IF_ERROR(slice_offset_operand->ReplaceAllUsesWith(loop_slice));
      TF_RETURN_IF_ERROR(computation->RemoveInstructionAndUnusedOperands(
          slice_offset_operand));
      TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(operand_to_replace));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
      changed = true;
    } else if (instruction->opcode() == HloOpcode::kWhile &&
               instruction->has_backend_config()) {
      TF_ASSIGN_OR_RETURN(
          bool while_changed,
          CanonicalizeSlices(instruction->called_computations()[0],
                             instruction));
      changed |= while_changed;
    }
  }
  return changed;
}

}  // namespace xla
