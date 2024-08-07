
#include "legate_mapper.h"
#include "legate_xla_c.h"
#include <core/mapping/mapping.h>
#include <core/utilities/typedefs.h>

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
  std::vector<mapping::StoreMapping> mappings;
  mappings.reserve(task.num_inputs() + task.num_outputs() +
                   task.num_reductions());

  const mapping::StoreTarget target = [&] {
    if (task.task_id() == legate::LocalTaskID{XLA_OFFLOAD_TASK}) {
      return mapping::StoreTarget::ZCMEM;
    }
    return options.front();
  }();

  auto append_mapping = [&](const auto &arrays) {
    for (auto &array : arrays) {
      auto stores = array.stores();
      for (auto &store : stores) {
        mappings.push_back(legate::mapping::StoreMapping::default_mapping(
            store, target, /*exact=*/true));
      }
    }
  };

  append_mapping(task.inputs());
  append_mapping(task.outputs());
  append_mapping(task.reductions());
  return mappings;
}

Scalar Mapper::tunable_value(TunableID tunable_id) { return Scalar(0); }

} // namespace legate_xla
