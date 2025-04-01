#ifndef XLA_PJRT_LEGATE_MPMD_HOIST_SHARD_MAP_REDUCE_H_
#define XLA_PJRT_LEGATE_MPMD_HOIST_SHARD_MAP_REDUCE_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

// Finds all all-reduce collectives that sum into
// a loop-carried summation in a while loop and moves
// the all-reduce out of the loop to execute once as a post-loop collective.
class MpmdHoistShardMapReduce : public HloModulePass {
 public:
  // The `partition` holds mesh information for each color.
  // For performance experiments, `remove_reduces` can be set to
  // true to remove the reduces instead of executing them.
  explicit MpmdHoistShardMapReduce(HloPartition* partition,
                                   bool remove_reduces = false)
      : partition_(partition), remove_reduces_(remove_reduces) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdHoistShardMapReduce() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-hoist-shard-map-reduce";
  }

 private:
  // Visit all instructions in the given `loop` (while instruction)
  // and convert all dots, reduces, and custom SPMD partitioning instructions
  // that are sharded over contraction dimensions into their manual
  // shard shapes with collectives inserted.
  absl::StatusOr<bool> VisitLoop(HloInstruction* loop,
                                 const InstructionProperties& properties,
                                 HloPassCleanup& cleanup);
  HloPartition* partition_;
  bool remove_reduces_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_HOIST_SHARD_MAP_REDUCE_H_