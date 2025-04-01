/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "xla/pjrt/legate/mpmd_input_output_buffer_alias.h"

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/errors.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/legate_sharding.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

absl::StatusOr<bool> MpmdInputOutputBufferAlias::Build(
    HloModule* module, absl::Span<const Shape> input_shapes,
    const Shape& output_shape, HloBufferDonorConfig* buffer_donor_config,
    bool match_temps) {
  bool changed = false;
  if (output_shape.is_dynamic()) {
    // Restrict dynamic shape input-output aliasing due to potential
    // dynamic shape size calculation mismatch.
    return false;
  }

  // Collects all buffer donors in a vector.
  struct DonorEntry {
    int64_t param_number;
    ShapeIndex index;
    int64_t shape_size;
  };
  std::vector<DonorEntry> donor_vectors;

  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      int64_t param_number = instruction->parameter_number();
      if (instruction->shape().IsTuple()) {
        return InvalidArgumentStrCat(
            "cannot perform MPMD input/output aliasing with arg tuples");
      }

      if (module->input_output_alias_config().ParameterHasAlias(param_number,
                                                                {})) {
        continue;
      }
      if (!buffer_donor_config->ParameterIsBufferDonor(param_number, {})) {
        continue;
      }

      Shape subshape = GetSpmdShape(instruction);
      if (match_temps == temporary_param_indices_.contains(param_number)) {
        VLOG(5) << "Pushing back donor {" << param_number << "} of size "
                << ShapeUtil::ByteSizeOf(subshape);
        if (!module->input_output_alias_config().ParameterHasAlias(param_number,
                                                                   {})) {
          donor_vectors.emplace_back(
              DonorEntry{param_number, {}, ShapeUtil::ByteSizeOf(subshape)});
        }
      }
    }
  }

  // Collects all buffer donees in a vector.
  struct DoneeEntry {
    ShapeIndex index;
    int64_t shape_size;
  };
  std::vector<DoneeEntry> donee_vectors;

  Shape spmd_shape =
      GetSpmdShape(module->entry_computation()->root_instruction());

  if (spmd_shape.IsTuple()) {
    for (int64_t index = 0; index < spmd_shape.tuple_shapes_size(); ++index) {
      if (match_temps == temporary_root_indices_.contains(index) &&
          !module->input_output_alias_config().OutputHasAlias({index})) {
        VLOG(5) << "Pushing back donee {" << index << "} of size "
                << ShapeUtil::ByteSizeOf(spmd_shape.tuple_shapes(index));
        donee_vectors.emplace_back(DoneeEntry{
            {index}, ShapeUtil::ByteSizeOf(spmd_shape.tuple_shapes(index))});
      }
    }
  } else {
    if (match_temps == temporary_root_indices_.contains(0) &&
        !module->input_output_alias_config().OutputHasAlias({})) {
      VLOG(5) << "Pushing back donee {} of size "
              << ShapeUtil::ByteSizeOf(spmd_shape);
      donee_vectors.emplace_back(
          DoneeEntry{{}, ShapeUtil::ByteSizeOf(spmd_shape)});
    }
  }

  // Sort donor and donees by their shape size in non-increasing order.
  absl::c_stable_sort(donor_vectors,
                      [](const DonorEntry& a, const DonorEntry& b) -> bool {
                        return a.shape_size > b.shape_size;
                      });
  absl::c_stable_sort(donee_vectors,
                      [](const DoneeEntry& a, const DoneeEntry& b) -> bool {
                        return a.shape_size > b.shape_size;
                      });

  // Match donors and donees with two pointers. The larger size a donee has, the
  // more prioritized the donee will get matched.
  int64_t donor_vector_index = 0;
  int64_t donee_vector_index = 0;
  while (donor_vector_index < donor_vectors.size() &&
         donee_vector_index < donee_vectors.size()) {
    const auto& donor = donor_vectors[donor_vector_index];
    const auto& donee = donee_vectors[donee_vector_index];

    if (donor.shape_size > donee.shape_size) {
      donor_vector_index += 1;
    } else if (donor.shape_size < donee.shape_size) {
      donee_vector_index += 1;
    } else {
      VLOG(3) << "Module " << module->name()
              << " matched donor=" << donor.param_number << " to "
              << donee.index << " with shape size " << donee.shape_size;
      // The current donor and donee match.
      TF_RETURN_IF_ERROR(module->input_output_alias_config().SetUpAlias(
          donee.index, donor.param_number, donor.index));
      TF_RETURN_IF_ERROR(buffer_donor_config->RemoveBufferDonor(
          donor.param_number, donor.index));
      donor_vector_index += 1;
      donee_vector_index += 1;
      changed = true;
    }
  }

  return changed;
}

absl::Status MpmdInputOutputBufferAlias::FindBestMatch(
    const ShapeIndex& output_index, HloInstruction* root, HloModule* module,
    HloInputOutputAliasConfig& alias_config, HloBufferDonorConfig* donors) {
  static constexpr int kMaxVisits = 1000;

  if (alias_config.OutputHasAlias(output_index)) {
    return absl::OkStatus();
  }

  if (output_index.size() == 1 &&
      temporary_root_indices_.contains(output_index.front())) {
    return absl::OkStatus();
  }

  bool tupled_args =
      module->entry_computation()->num_parameters() == 1 &&
      module->entry_computation()->parameter_instruction(0)->shape().IsTuple();
  if (tupled_args) {
    return InvalidArgumentStrCat(
        "Legate buffer donation does not suppoort tupled args");
  }

  auto is_donor_parameter = [&](HloInstruction* operand) {
    if (operand->opcode() == HloOpcode::kParameter &&
        !temporary_param_indices_.contains(operand->parameter_number()) &&
        !alias_config.ParameterHasAlias(operand->parameter_number(), {})) {
      return donors->ParameterIsBufferDonor(operand->parameter_number(), {});
    }
    return false;
  };

  if (is_donor_parameter(root)) {
    return alias_config.SetUpAlias(output_index, root->parameter_number(), {});
  }

  std::deque<HloInstruction*> to_visit{root->operands().begin(),
                                       root->operands().end()};
  int visit_count = 0;
  while (!to_visit.empty() && visit_count < kMaxVisits) {
    auto* operand = to_visit.front();
    to_visit.pop_front();
    if (is_donor_parameter(operand) && operand->shape() == root->shape()) {
      VLOG(3) << "found parameter-root alias match between " << root->name()
              << " and " << operand->name() << " for pair "
              << operand->parameter_number() << " -> " << output_index;
      return alias_config.SetUpAlias(output_index, operand->parameter_number(),
                                     {});
    }
    // keep looking
    for (auto* next : operand->operands()) {
      to_visit.push_back(next);
    }
    ++visit_count;
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> MpmdInputOutputBufferAlias::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // We exactly follow HloInputOutputAliasConfig::Verify to create input_shapes
  // and output_shape.
  const auto& entry_computation_layout = module->entry_computation_layout();
  std::vector<Shape> input_shapes;
  for (int64_t i = 0; i < module->entry_computation()->num_parameters(); ++i) {
    input_shapes.push_back(entry_computation_layout.parameter_shape(i));
  }
  const Shape& output_shape = entry_computation_layout.result_shape();

  HloBufferDonorConfig* buffer_donor_config = &module->buffer_donor_config();

  std::vector<HloInstruction*> to_vist;

  if (module->entry_computation()->root_instruction()->shape().IsTuple()) {
    int64_t root_number = 0;
    for (auto* operand :
         module->entry_computation()->root_instruction()->operands()) {
      TF_RETURN_IF_ERROR(FindBestMatch(
          ShapeIndex({root_number}), operand, module,
          module->input_output_alias_config(), buffer_donor_config));
      ++root_number;
    }
  } else {
    TF_RETURN_IF_ERROR(FindBestMatch(
        ShapeIndex({}), module->entry_computation()->root_instruction(), module,
        module->input_output_alias_config(), buffer_donor_config));
  }

  // Run matcher twice
  // First to match params/roots
  // Second to match temp intermediates
  TF_ASSIGN_OR_RETURN(bool matched_params,
                      Build(module, input_shapes, output_shape,
                            buffer_donor_config, /*match_temps=*/true));
  TF_ASSIGN_OR_RETURN(bool matched_temps,
                      Build(module, input_shapes, output_shape,
                            buffer_donor_config, /*match_temps=*/false));

  TF_RETURN_IF_ERROR(module->input_output_alias_config().Verify(
      *module,
      [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape); }));

  return matched_params || matched_temps;
}

}  // namespace xla
