/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_logical_sharding_propagation.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/multimesh/logical_sharding_context.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

absl::StatusOr<bool> MpmdLogicalShardingPropagation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloComputation*> to_visit = {module->entry_computation()};
  bool changed = false;
  auto properties = InstructionProperties::Create(module);
  HloPassCleanup cleanup;
  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    VLOG(5) << "visiting computation " << computation->name();

    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->IsCustomCall("AutoSharding")) {
        auto* operand = instruction->mutable_operand(0);
        std::string axis_json = instruction->raw_backend_config_string();
        TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(operand));
        cleanup.RemoveInstruction(instruction);
        properties.ForEachAlias(operand, [&](HloInstruction* i) {
          VLOG(5) << "logical sharding applied to " << i->name() << " "
                  << i->shape() << ": " << axis_json;
          AssignAxes(i, axis_json);
        });
        VLOG(5) << "logical sharding applied to " << operand->name() << " "
                << operand->shape() << ": " << axis_json;

        auto& instruction_props = properties.Get(instruction);
        if (instruction_props.elementwise_connected_to_entry_output) {
          // make sure any connected outputs are given a matching autosharding
          AssignAxes(instruction_props.elementwise_connected_to_entry_output,
                     axis_json);
        }

        AssignAxes(operand, std::move(axis_json));
        changed = true;
      } else if (instruction->opcode() == HloOpcode::kCall) {
        auto* called_computatation = instruction->called_computations()[0];
        for (int64_t index = 0; index < instruction->operand_count();
             ++instruction) {
          auto axes = GetAxesString(instruction->operand(index));
          if (axes.has_value()) {
            auto* param = called_computatation->parameter_instruction(index);
            VLOG(5) << "logical sharding applied to call param "
                    << param->name() << " " << param->shape() << ": " << *axes;
            AssignAxes(param, *std::move(axes));
          }
        }
        to_visit.push_back(called_computatation);
      } else if (instruction->opcode() == HloOpcode::kTuple &&
                 instruction->users().size() == 1) {
        auto* user_tuple = [&] {
          if (instruction->users().front()->opcode() == HloOpcode::kWhile) {
            return instruction->users()
                .front()
                ->called_computations()[0]
                ->parameter_instruction(0);
          } else if (instruction->users().front()->opcode() ==
                     HloOpcode::kOptimizationBarrier) {
            return instruction->users().front();
          }
          return (HloInstruction*)nullptr;
        }();

        if (user_tuple) {
          for (auto* user : user_tuple->users()) {
            auto* matching_operand = instruction->operand(user->tuple_index());
            auto axes = GetAxesString(matching_operand);
            if (axes.has_value()) {
              VLOG(5) << "logical sharding applied to tuple user "
                      << user->name() << " " << user->shape() << ": " << *axes;
              AssignAxes(user, *std::move(axes));
            }
          }
        }
      } else if (instruction->opcode() == HloOpcode::kDot &&
                 !HasAssignedAxes(instruction)) {
        // I would prefer not to do dot logical sharding this way and let GSPMD
        // do the sharding propagation, but GSPMD makes strange decisions on dot
        // sharding when the contracting dimension is sharded. This is needed to
        // avoid scenarios in TE when the dot operands are given specific
        // sharding constraints and TE expects the dot to have a particular
        // output sharding
        auto* lhs = instruction->operand(0);
        auto* rhs = instruction->operand(1);
        auto lhs_axes = GetAxes(lhs);
        auto rhs_axes = GetAxes(rhs);
        if (lhs_axes.has_value() && rhs_axes.has_value()) {
          VLOG(5) << "dot " << instruction->name()
                  << " without axes has LHS and RHS with axes assigned";
          LogicalShardingAxes dot_axes;
          dot_axes.axes.resize(instruction->shape().dimensions_size());
          HloDotInstruction* dot = static_cast<HloDotInstruction*>(instruction);
          absl::flat_hash_set<int64_t> rhs_cxn_and_batch_dims;
          absl::flat_hash_set<int64_t> lhs_cxn_and_batch_dims;
          absl::flat_hash_set<std::string> axes_already_assigned;
          for (int64_t dim :
               dot->dot_dimension_numbers().lhs_contracting_dimensions()) {
            lhs_cxn_and_batch_dims.insert(dim);
          }
          for (int64_t dim :
               dot->dot_dimension_numbers().rhs_contracting_dimensions()) {
            rhs_cxn_and_batch_dims.insert(dim);
          }

          auto maybe_add_axes_at_dim = [&dot_axes, &axes_already_assigned](
                                           int dot_dim, int operand_dim,
                                           const LogicalShardingAxes& axes) {
            if (dot_axes.axes[dot_dim]
                    .empty()) {  // batch dimensions can come from both
              for (const auto& axis_name : axes.axes[operand_dim]) {
                if (!axes_already_assigned.contains(axis_name)) {
                  dot_axes.axes[dot_dim].push_back(axis_name);
                  axes_already_assigned.insert(axis_name);
                }
              }
            }
          };

          // there is no obvious reason to prioritize the RHS names over the LHS
          // names for now, we prefer RHS names because TE relies on that being
          // the case TE has incomplete sharding constraints and generally seems
          // to rely on RHS shardings being kept in the output

          int64_t dot_dim = 0;
          for (int64_t batch_dim :
               dot->dot_dimension_numbers().rhs_batch_dimensions()) {
            rhs_cxn_and_batch_dims.insert(batch_dim);
            maybe_add_axes_at_dim(dot_dim, batch_dim, *rhs_axes);
            ++dot_dim;
          }

          dot_dim =
              lhs->shape().dimensions_size() -
              dot->dot_dimension_numbers().lhs_contracting_dimensions_size();
          for (int64_t dim = 0; dim < rhs->shape().dimensions_size(); ++dim) {
            if (!rhs_cxn_and_batch_dims.contains(dim)) {
              maybe_add_axes_at_dim(dot_dim, dim, *rhs_axes);
              ++dot_dim;
            }
          }

          dot_dim = 0;
          for (int64_t batch_dim :
               dot->dot_dimension_numbers().lhs_batch_dimensions()) {
            lhs_cxn_and_batch_dims.insert(batch_dim);
            maybe_add_axes_at_dim(dot_dim, batch_dim, *lhs_axes);
            ++dot_dim;
          }

          for (int64_t dim = 0; dim < lhs->shape().dimensions_size(); ++dim) {
            if (!lhs_cxn_and_batch_dims.contains(dim)) {
              maybe_add_axes_at_dim(dot_dim, dim, *lhs_axes);
              ++dot_dim;
            }
          }

          AssignAxes(instruction, dot_axes);
        }
      }
      for (auto* subcomp : instruction->called_computations()) {
        to_visit.push_back(subcomp);
      }
    }
  }

  auto* root = module->entry_computation()->root_instruction();
  module->input_output_alias_config().ForEachAlias(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) {
        auto* parameter = module->entry_computation()->parameter_instruction(
            alias.parameter_number);
        auto* output = [&] {
          if (root->opcode() == HloOpcode::kTuple) {
            return root->mutable_operand(output_index.front());
          }
          return root;
        }();

        // if axes have been assigned to an aliased input or output
        // share autosharding between them
        auto axes = GetAxesString(parameter);
        if (axes.has_value() && !HasAssignedAxes(output)) {
          properties.ForEachBackwardAliasAndSelf(
              output, [&](HloInstruction* i) { AssignAxes(i, *axes); });
        } else {
          axes = GetAxesString(output);
          if (axes.has_value()) {
            properties.ForEachForwardAliasAndSelf(
                parameter, [&](HloInstruction* i) { AssignAxes(i, *axes); });
          }
        }
      });

  TF_RETURN_IF_ERROR(cleanup.CleanUp());
  return changed;
}

}  // namespace xla
