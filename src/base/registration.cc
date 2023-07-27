#include "legate.h"
#include "legate_mapper.h"
#include "xla_task.h"

namespace legate_xla {

namespace {
static constexpr char library_name[] = "legate.xla";
}

extern void initialize_runtime_and_context(legate::Runtime *runtime,
                                           legate::Library library);
/*static*/ void registration_callback() {
  legate::ResourceConfig config;
  config.max_tasks = 64;
  config.max_projections = 0;
  // We register one sharding functor for each new projection functor
  config.max_shardings = 0;
  config.max_reduction_ops = 0;

  auto runtime = legate::Runtime::get_runtime();

  auto context =
      runtime->create_library(library_name, config, std::make_unique<Mapper>());

  initialize_runtime_and_context(runtime, context);

  Registry::get_registrar().register_all_tasks(context);
}

} // namespace legate_xla

void legate_xla_perform_registration() {
  legate::Core::perform_registration<&legate_xla::registration_callback>();
}