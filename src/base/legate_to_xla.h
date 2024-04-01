#include "legate_xla_common.h"
#include <memory>
#include <optional>
#include <string>

extern "C" void CompileHloModuleFromFile(const std::string &hlo_file,
                                         const std::string &platform_name,
                                         int replica_count, int num_partitions,
                                         bool erase_sharding, bool autoshard,
                                         std::optional<int64_t> device_mem);

struct PJRT_Api;
extern "C" const PJRT_Api *GetLegatePjrtApi();

extern "C" void ShutdownLegateClient();