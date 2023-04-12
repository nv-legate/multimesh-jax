/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_REPEATED_OUTPUT_COPY_H_
#define XLA_PJRT_MULTIMESH_MPMD_REPEATED_OUTPUT_COPY_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/util.h"

namespace xla {

// Creates an explicit copy of all outputs (root tuple operands)
// that are repeated. Even if an output is passes multiple times
// as a root tuple operand, distinct copies in different buffers
// need to be returned. To simplify the analysis of inputs/outputs
// in later passes, this creates a copy of each output
// and passes the copy as an operand to the root tuple.
class MpmdRepeatedOutputCopy : public HloModulePass {
 public:
  // The `partition` object contains the mapping from partition color
  // to the assigned submesh.
  explicit MpmdRepeatedOutputCopy(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdRepeatedOutputCopy() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-repeated-output-copy";
  }

 private:
  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_REPEATED_OUTPUT_COPY_H_
