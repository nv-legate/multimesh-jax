#pragma once

enum XlaOpCode{
  XLA_COMPILE_TASK,
  XLA_EXECUTE_TASK,
  HLO_PROTOTYPE_LOAD,
  HLO_PROTOTYPE_EXECUTE,
  HLO_PROTOTYPE_DISTRIBUTED_INIT,
  HLO_PROTOTYPE_DISTRIBUTED_SHUTDOWN,
  HLO_PROTOTYPE_FILL,
};

#ifdef __cplusplus
extern "C" {
#endif

void legate_xla_perform_registration();

#ifdef __cplusplus
}
#endif