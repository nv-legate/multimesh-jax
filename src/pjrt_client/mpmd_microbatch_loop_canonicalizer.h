#ifndef XLA_PJRT_LEGATE_MPMD_MICROBATCH_LOOP_CANONICALIZER_H_
#define XLA_PJRT_LEGATE_MPMD_MICROBATCH_LOOP_CANONICALIZER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"

namespace xla {

// Loops through all Legate custom calls defining the microbatching
// and loop schedules and records the information. All
// Legate custom calls for microbatching are removed during the pass.
class MpmdMicrobatchLoopCanonicalizer : public HloModulePass {
 public:
  // The `partition` object contains the mapping from partition color
  // to the assigned submesh.
  explicit MpmdMicrobatchLoopCanonicalizer(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdMicrobatchLoopCanonicalizer() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-microbatch-while-loop-inliner";
  }

 private:
  absl::StatusOr<bool> CanonicalizeSlices(HloComputation* computation,
                                          HloInstruction* parent_loop);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_MICROBATCH_LOOP_CANONICALIZER_H_