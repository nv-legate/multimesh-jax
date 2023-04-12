
/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_INPUT_OUTPUT_BUFFER_ALIAS_H_
#define XLA_PJRT_MULTIMESH_MPMD_INPUT_OUTPUT_BUFFER_ALIAS_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

// Matches buffer donor parameters with a matching output.
// This is a generalization of the standard alias matching
// passes that separately considers parameters, roots,
// and MPMD temporaries.  MPMD temporaries cannot be
// paired with parameters or outputs.
class MpmdInputOutputBufferAlias : public HloModulePass {
 public:
  // The `temporary_param_indices` given the parameter numbers
  // that correspond to MPMD temporaries. The set of
  // `temporary_root_indices` gives the root tuple
  // operand indices (output numbers) that correspond to temporaries.
  explicit MpmdInputOutputBufferAlias(
      absl::flat_hash_set<int64_t> temporary_param_indices,
      absl::flat_hash_set<int64_t> temporary_root_indices)
      : temporary_param_indices_(std::move(temporary_param_indices)),
        temporary_root_indices_(std::move(temporary_root_indices)) {}

  ~MpmdInputOutputBufferAlias() override = default;

  absl::string_view name() const override {
    return "mpmd_input_output_buffer_alias.h";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::flat_hash_set<int64_t> temporary_root_indices_;
  absl::flat_hash_set<int64_t> temporary_param_indices_;

  // For a given `output_index` of the `root` instruction (tuple or single
  // instruction), this loops through all parameters of the `module` entry
  // computation and uses a matching heuristic to find the parameter that best
  // matches the output. The matching will consider shape, sharding, and use
  // distance (affinity) between the parameter and the root.
  absl::Status FindBestMatch(const ShapeIndex& output_index,
                             HloInstruction* root, HloModule* module,
                             HloInputOutputAliasConfig& alias_config,
                             HloBufferDonorConfig* donors);

  absl::StatusOr<bool> Build(HloModule* module,
                             absl::Span<const Shape> input_shapes,
                             const Shape& output_shape,
                             HloBufferDonorConfig* buffer_donor_config,
                             bool match_temps);
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_INPUT_OUTPUT_BUFFER_ALIAS_H_
