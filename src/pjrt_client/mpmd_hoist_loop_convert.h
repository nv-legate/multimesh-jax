/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_MPMD_HOIST_LOOP_CONVERT_H_
#define XLA_PJRT_LEGATE_MPMD_HOIST_LOOP_CONVERT_H_

#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"

namespace xla {

// Whenver a paramter convert is repeated inside a loop, the pass
// hoists the convert out of the loop to avoid the recomputation
// overhead. The matching pattern is:
//
// A = parameter
// while:
//   B = convert(A)
//   C = instruction(B, ...)
//
// this pass hoists the convert to
//
// A = parameter
// B = convert(A)
// while:
//   C = instruction(B, ...)
class MpmdHoistLoopConvert : public HloModulePass {
 public:
  // The `partition` holds mesh information for each color.
  // For performance experiments, `zero_out_arguments` can be set to true
  // instead converting the actual parameter values.
  explicit MpmdHoistLoopConvert(HloPartition* partition,
                                bool zero_out_arguments = false)
      : partition_(partition), zero_out_arguments_(zero_out_arguments) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdHoistLoopConvert() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "mpmd-hoist-loop-convert"; }

 private:
  // Clones the `computation` in the given `module` where
  // the computation has a single parameter with tuple shape.
  // The arg tuple's shape will be replaced with `arg_tuple_shapes`
  // and the new shapes propagated to the get-tuple-element
  // users of the arg tuple.
  absl::StatusOr<HloComputation*> CloneArgTupleComputation(
      HloComputation* computation, HloModule* module,
      const std::vector<Shape>& arg_tuple_shapes);

  HloPartition* partition_;
  bool zero_out_arguments_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_HOIST_LOOP_CONVERT_H_
