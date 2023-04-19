#include "core/mapping/base_mapper.h"
#include "legate.h"

namespace legate_xla {

// Legate XLA mapper
class Mapper : public legate::mapping::LegateMapper {
 public:
  Mapper();
  virtual ~Mapper(void) {}

 private:
  Mapper(const Mapper& rhs) = delete;
  Mapper& operator=(const Mapper& rhs) = delete;

  // Legate mapping functions
 public:
  void set_machine(
      const legate::mapping::MachineQueryInterface* machine) override;
  legate::mapping::TaskTarget task_target(
      const legate::mapping::Task& task,
      const std::vector<legate::mapping::TaskTarget>& options) override;
  std::vector<legate::mapping::StoreMapping> store_mappings(
      const legate::mapping::Task& task,
      const std::vector<legate::mapping::StoreTarget>& options) override;
  legate::Scalar tunable_value(legate::TunableID tunable_id) override;
};

}