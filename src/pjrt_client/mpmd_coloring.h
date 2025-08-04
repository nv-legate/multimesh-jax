/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_COLORING_H_
#define XLA_PJRT_MULTIMESH_MPMD_COLORING_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {

// Propagates a coloring to all non-trivial operations in the HLO module
// to create a partition of all operations based on user annotations.
class MpmdColoring : public HloModulePass {
 public:
  enum class ColorPropagationPriority {
    kWeight,
    kDepth,
    kTopological,
  };

  static ColorPropagationPriority GetColorPropagationPriorityFromString(
      absl::string_view priority_str);

  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdColoring(HloPartition* partition,
                        ColorPropagationPriority color_propagation_priority =
                            ColorPropagationPriority::kWeight)
      : partition_(partition),
        color_propagation_priority_(color_propagation_priority) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdColoring() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "mpmd-coloring"; }

 private:
  using FilterVisitFn = absl::FunctionRef<bool(const HloInstruction*)>;

  // Propagate partition colors through the given `computation`.
  // baed on the full set of module `properties`. Propagation
  // will occur to/from an instruction when `if_visit` function
  // returns true.  The filter may limit partition coloring to
  // only occur across elementwise, skip tuples, etc.
  // The coloring is identified as belonging to a `microbatch_loop`.
  absl::StatusOr<bool> PropagateIf(HloComputation* computation,
                                   const InstructionProperties& properties,
                                   FilterVisitFn if_visit);

  absl::StatusOr<bool> PropagateLoopColorDepth(
      HloComputation* computation, const InstructionProperties& properties,
      FilterVisitFn if_visit);

  bool PropagateDirectionally(
      const std::vector<HloInstruction*>& postorder, const std::string& color,
      const InstructionProperties& properties, FilterVisitFn if_visit,
      const absl::flat_hash_set<HloInstruction*>& fixed_colored_instructions,
      bool propagate_forward);

  // For a given `instruction` based on the global module `properties`,
  // choose a best possible partition color from all the users and
  // operands of the instruction. Partition colors will be propagated
  // for users/operands when `if_visit` returns true on the instruction.
  // The lookup map `depth` gives the minimum depth of the instruction
  // in the graph.
  bool PropagateFromUsersAndOperands(
      HloInstruction* instruction, const InstructionProperties& properties,
      FilterVisitFn if_visit,
      const absl::flat_hash_map<const HloInstruction*, int64_t>& depth,
      const absl::flat_hash_map<const HloInstruction*, int64_t>&
          topological_index);

  HloPartition* partition_;
  ColorPropagationPriority color_propagation_priority_;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MPMD_COLORING_H_
