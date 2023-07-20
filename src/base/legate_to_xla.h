#include "legate_xla_common.h"
#include <memory>
#include <string>

extern "C" std::unique_ptr<legate_xla::LegateCompiler>
GetLegateCompilerFromHloProtoText(const std::string &hlo_string,
                                  const std::string &platform_name,
                                  int replica_count, int num_partitions);

extern "C" std::unique_ptr<legate_xla::LegateCompiler>
GetLegateCompilerFromHloProtoFile(const std::string &hlo_file,
                                  const std::string &platform_name,
                                  int replica_count, int num_partitions);

extern "C" bool InitDistributedRuntime(const std::string &coordinator_addr,
                                       int coordinator_port, int num_procs,
                                       int proc_id, int gpus_per_proc);

extern "C" bool ShutdownDistributedRuntime();

struct PJRT_Api;
extern "C" const PJRT_Api *GetLegatePjrtApi();