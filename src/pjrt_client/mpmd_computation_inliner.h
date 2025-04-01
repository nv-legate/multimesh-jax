#ifndef XLA_PJRT_LEGATE_MPMD_COMPUTATION_INLINER_H_
#define XLA_PJRT_LEGATE_MPMD_COMPUTATION_INLINER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"

namespace xla {

// Reverses MpmdComputationGrouper and puts the module back in "flat"
// form. This simplifies use analysis since operands/users of an instruction
// will be within the same computation rather than parameters/outputs
// of a call.
class MpmdComputationInliner : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdComputationInliner(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdComputationInliner() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "mpmd-computation-inliner"; }

 private:
  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_COMPUTATION_INLINER_H_