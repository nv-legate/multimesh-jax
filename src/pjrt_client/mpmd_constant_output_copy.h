#ifndef XLA_PJRT_LEGATE_MPMD_CONSTANT_OUTPUT_COPY_H_
#define XLA_PJRT_LEGATE_MPMD_CONSTANT_OUTPUT_COPY_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"

namespace xla {

// Simplifies analysis of the module by changing
// any outputs that are a constant to a copy of the constant.
// Constants need to be replicated across partitions,
// which becomes more complicated when constants are root tuple
// operands. This simplifies constant replication later.
class MpmdConstantOutputCopy : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdConstantOutputCopy(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdConstantOutputCopy() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "mpmd-constant-output"; }

 private:
  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_CONSTANT_OUTPUT_COPY_H_