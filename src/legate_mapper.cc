
#include "legate_mapper.h"
#include "legate_xla_c.h"

using namespace legate;

namespace legate_xla {

Mapper::Mapper(){};

void Mapper::set_machine(const legate::mapping::MachineQueryInterface *machine){
    // FIXME
};

mapping::TaskTarget
Mapper::task_target(const mapping::Task &task,
                    const std::vector<mapping::TaskTarget> &options) {
  return *options.begin();
}

std::vector<mapping::StoreMapping>
Mapper::store_mappings(const mapping::Task &task,
                       const std::vector<mapping::StoreTarget> &options) {
  return {};
}

Scalar Mapper::tunable_value(TunableID tunable_id) {
  LEGATE_ABORT;
  return Scalar();
}

} // namespace legate_xla