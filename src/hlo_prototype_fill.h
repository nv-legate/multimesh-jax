#pragma once

#include "legate_xla_c.h"
#include "xla_task.h"

namespace legate_xla {

class HLOFillTask : public XlaTask<HLOFillTask> {
public:
  static const int TASK_ID = HLO_PROTOTYPE_FILL;

public:
  static void cpu_variant(legate::TaskContext &context);
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext &context);
#endif
};

} // namespace legate_xla