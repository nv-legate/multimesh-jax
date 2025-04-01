#ifndef XLA_PJRT_LEGATE_MPMD_INSTRUCTION_DELAY_RECOLOR_H_
#define XLA_PJRT_LEGATE_MPMD_INSTRUCTION_DELAY_RECOLOR_H_

#include <cstdint>
#include <vector>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_utils.h"
namespace xla {

// Reassigns the partition color of instructions to schedule them
// later.  If an instruction that is not part of the critical path
// is iniitally scheduled early, it may delay other instructions
// on the critical path unnecessarily. This moves instructions
// as late as possible by find partition colors later in the schedule
// and reassigning instructions, when allowed and beneficial,
// to that partition color. Consider the case:
//
// A = op(params) color (red)
// B = op(params) color (red)
// C = op(A) color(blue)
// D = op(C) color(green)
// E = op(D) color(orange)
//
// when orange/red occur on the same mesh, this will reassign
// instruction B so that it no longer delays the A->C->D->E
// critical path.
//
// A = op(params) color (red)
// C = op(A) color(blue)
// D = op(C) color(green)
// E = op(D) color(orange)
// B = op(params) color (orange)
class MpmdInstructionDelayRecolor : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdInstructionDelayRecolor(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdInstructionDelayRecolor() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-instruction-delay-recolor";
  }

 private:
  absl::StatusOr<std::vector<HloInstruction*>> MaybeRecolorOutput(
      HloInstruction* output, HloInstruction* call, HloComputation* parent,
      const absl::flat_hash_map<HloInstruction*, int64_t>& depth,
      const InstructionProperties& properties);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_INSTRUCTION_DELAY_RECOLOR_H_