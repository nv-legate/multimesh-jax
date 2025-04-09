#ifndef XLA_PJRT_LEGATE_COLOR_DFS_H_
#define XLA_PJRT_LEGATE_COLOR_DFS_H_

#include "absl/status/status.h"
#include "src/zuku/mesh.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

std::vector<HloInstruction*> ColorSortedPostorder(HloComputation* computation);

}  // namespace xla

#endif
