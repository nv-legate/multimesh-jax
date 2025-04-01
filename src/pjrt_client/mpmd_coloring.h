#ifndef XLA_PJRT_LEGATE_MPMD_COLORING_H_
#define XLA_PJRT_LEGATE_MPMD_COLORING_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

// Assigns a coloring to all non-trivial operations in the HLO module
// to create a partition of all operations based on user annotations.
// Partitions (colors) are assigned to instructions as a frontend attribute.
class MpmdColoring : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdColoring(HloPartition* partition) : partition_(partition) {}

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
                                   FilterVisitFn if_visit,
                                   bool microbatch_loop);

  // For a given `instruction` based on the global module `properties`,
  // choose a best possible partition color from all the users and
  // operands of the instruction. Partition colors will be propagated
  // for users/operands when `if_visit` returns true on the instruction.
  // The lookup map `depth` gives the minimum depth of the instruction
  // in the graph.
  bool PropagateFromUsersAndOperands(
      HloInstruction* instruction, const InstructionProperties& properties,
      FilterVisitFn if_visit,
      const absl::flat_hash_map<const HloInstruction*, int64_t>& depth);

  // For a given `computation` and global module `properties`, compute
  // a partition color for the isntruction if it has been directly
  // assigned by the user by either an explicit task block or
  // an op_name matcher.
  absl::Status ComputeAssignedColors(HloComputation* computation,
                                     const InstructionProperties& properties);

  // For a given `computation`, iterate through and inline all explicit
  // Legate task blocks. This is equivalent to running the CallInliner
  // on the computation and assigning the scope color as a frontend attribute.
  absl::StatusOr<bool> InlineExplicitTasks(HloComputation* computation);

  // Assign a color to the given tuple `instruction`, if uniformly colored.
  absl::Status ColorTuple(HloInstruction* instruction);

  // If the given `instruction`, has not been assigned a color based on
  // user annotation, but the instruction does have an explicit sharding
  // then assign the instruction to a partition over the sharding devices
  // since any instruction explicitly assigned to a given device mesh
  // must be assigned a specific partition color and cannot be freely
  // assigned anywhere.
  absl::StatusOr<std::optional<std::string>> CheckForExplicitShardingColor(
      HloInstruction* instruction, const InstructionProperties& properties);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_COLORING_H_