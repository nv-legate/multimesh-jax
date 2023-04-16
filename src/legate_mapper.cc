
#include "legate_mapper.h"

using namespace legate;

#if 1 //ndef LEGATE_XLA_PYTHON_PROTOTYPE

namespace legate_xla {

Mapper::Mapper(){};

void Mapper::set_machine(
    const legate::mapping::MachineQueryInterface* machine){
    // FIXME
};

mapping::TaskTarget Mapper::task_target(
    const mapping::Task& task,
    const std::vector<mapping::TaskTarget>& options) {
  return *options.begin();
}

std::vector<mapping::StoreMapping> Mapper::store_mappings(
    const mapping::Task& task,
    const std::vector<mapping::StoreTarget>& options) {
  return {};
}

Scalar Mapper::tunable_value(TunableID tunable_id) {
  LEGATE_ABORT;
  return Scalar();
}

#else

LLMMapper::LLMMapper(Legion::Runtime* rt, Legion::Machine m, const LibraryContext& ctx)
  : BaseMapper(rt,
               m,
               ctx,
               {.default_contiguous           = false,
                .single_store_per_mapping     = true,
                .disjoint_instances           = true,
                .force_alloc_on_single_fields = false,
                .default_inorder              = false})
{
}

TaskTarget LLMMapper::task_target(const Task& task, const std::vector<TaskTarget>& options)
{
  return *options.begin();
}

void LLMMapper::map_task(const Legion::Mapping::MapperContext ctx,
                         const Legion::Task& task,
                         const MapTaskInput& input,
                         MapTaskOutput& output)
{
  // log_llm.debug() << "Starting map task " << task.get_provenance_string();
  BaseMapper::map_task(ctx, task, input, output);
  // log_llm.debug() << "Finish map task " << task.get_provenance_string();
}

void LLMMapper::select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                                    const SelectMappingInput& input,
                                    SelectMappingOutput& output)
{
  BaseMapper::select_tasks_to_map(ctx, input, output);
  // for (auto& task : output.map_tasks) {
  //   log_llm.debug() << "Selected task to map: " << task->get_provenance_string();
  // }
}

Scalar LLMMapper::tunable_value(TunableID tunable_id)
{
  switch (tunable_id) {
    case LLM_TUNABLE_NUM_GPUS: {
      int32_t num_gpus = machine->gpus().size() * machine->total_nodes;
      return Scalar(num_gpus);
    }
    case LLM_TUNABLE_NUM_PROCS: {
      int32_t num_procs = 0;
      if (!machine->gpus().empty())
        num_procs = machine->gpus().size() * machine->total_nodes;
      else if (!machine->omps().empty())
        num_procs = machine->omps().size() * machine->total_nodes;
      else
        num_procs = machine->cpus().size() * machine->total_nodes;
      return Scalar(num_procs);
    }
    default: break;
  }
  LEGATE_ABORT;  // unknown tunable value
}

#endif

}