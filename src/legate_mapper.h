#include "core/mapping/base_mapper.h"
#include "legate.h"

#if 1 //ndef LEGATE_XLA_PYTHON_PROTOTYPE

namespace legate_xla {

// Legate JAX mapper
class Mapper : public legate::mapping::LegateMapper {
 public:
  // LegateJAXMapper(Legion::Runtime* rt, Legion::Machine machine,
  //                 const legate::LibraryContext& context);
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

#else

class LLMMapper : public legate::mapping::BaseMapper {
 public:
  LLMMapper(Legion::Runtime* rt, Legion::Machine machine, const legate::LibraryContext& context);
  virtual ~LLMMapper(void) {}

 private:
  LLMMapper(const LLMMapper& rhs)            = delete;
  LLMMapper& operator=(const LLMMapper& rhs) = delete;

  // Legate mapping functions
 public:
  bool is_pure() const override { return true; }

  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override;

  legate::Scalar tunable_value(legate::TunableID tunable_id) override;

  void map_task(const Legion::Mapping::MapperContext ctx,
                const Legion::Task& task,
                const MapTaskInput& input,
                MapTaskOutput& output) override;

  void select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                           const SelectMappingInput& input,
                           SelectMappingOutput& output) override;
};

#endif

}