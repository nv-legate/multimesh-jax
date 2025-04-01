#ifndef XLA_PJRT_LEGATE_MPMD_LOGICAL_TO_GSPMD_SHARDING_H_
#define XLA_PJRT_LEGATE_MPMD_LOGICAL_TO_GSPMD_SHARDING_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

// Translates the logical sharding annotations with named axes
// to GSPMD shardings on a physical device mesh based on the mesh
// assigned to a particular partition in the graph.
class MpmdLogicalToGSPMDSharding : public HloModulePass {
 public:
  // The `partition` object contains the mapping from partition color
  // to the assigned submesh.
  explicit MpmdLogicalToGSPMDSharding(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdLogicalToGSPMDSharding() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-logical-to-gspmd-sharding";
  }

 private:
  absl::StatusOr<bool> ApplySharding(HloInstruction* instruction,
                                     const InstructionProperties& properties);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_LOGICAL_TO_GSPMD_SHARDING_H_
