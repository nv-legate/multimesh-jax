#ifndef XLA_PJRT_LEGATE_LOOP_SCHEDULER_H_
#define XLA_PJRT_LEGATE_LOOP_SCHEDULER_H_

#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_loop.h"

namespace xla {

absl::StatusOr<std::vector<HloInstruction*>> ScheduleLoops(
    const HloPartition& partition, const LoopConfig& config,
    const std::vector<std::vector<HloInstruction*>>& tasks);

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_LOOP_SCHEDULER_H_