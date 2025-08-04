/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_COMPUTATION_FUSION_H_
#define XLA_PJRT_MULTIMESH_MPMD_COMPUTATION_FUSION_H_

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/util.h"

namespace xla {

absl::Status ClearPredecessors(HloInstruction* instruction);

absl::Status ClearControlDependencies(HloInstruction* instruction);

// Fuse the list of `calls` into a single computation.
// If `fused_computation` is null, a new computation is generated
// that is equivalent to the fused calls. If `fused_computation` is
// non-null, no new computation is generated and `fused_computation`
// is used directl to create the new call. No error checking is done
// on `fused_computation` to ensure that it is actually equivalaent
// to creating a new fusion.
absl::StatusOr<HloInstruction*> FuseCalls(
    absl::Span<HloInstruction*> calls,
    HloComputation* fused_computation = nullptr);

// Finds computations with the same color or same device mesh
// and fuses them into a single computation, creating a new set
// of parameter instructions and new set of outputs.
class MpmdComputationFusion : public HloModulePass {
 public:
  enum FusionType { kMatchingDevices, kMatchingColor, kOriginalColor };

  // The `partition` object containing the mapping from partition color
  // to the assigned submesh. Different fusions will be allowed
  // based on the `type` to either fuse only identical partition colors
  // or to fuse any partition colos on the ame devices.
  // To increase parallelism or decrease temp buffe required,
  // `only_fuse_loop_tasks` can be set to true.
  explicit MpmdComputationFusion(HloPartition* partition, FusionType type,
                                 bool only_fuse_loop_tasks)
      : partition_(partition),
        type_(type),
        only_fuse_loop_tasks_(only_fuse_loop_tasks) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdComputationFusion() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "mpmd-computation-fusion"; }

 private:
  // Recursively visit all instructions in the given `computation` and
  // fuse any matching computations. Returns true if any fusions occurred
  // in this computation or any subcomputations.
  absl::StatusOr<bool> Visit(HloComputation* computation);

  // Returns true the if `lhs` and `rhs` match according to the
  // given FusionType for the HLO pass and can be fused.
  bool FusionMatch(const HloInstruction* lhs, const HloInstruction* rhs);

  // Fuse the computations from the `producer` and `consumer`
  // call instruction inside the `parent` computation.
  absl::Status FuseComputations(HloComputation* parent,
                                absl::Span<HloInstruction*> calls);

  HloPartition* partition_;
  FusionType type_;
  bool only_fuse_loop_tasks_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_COMPUTATION_FUSION_H_
