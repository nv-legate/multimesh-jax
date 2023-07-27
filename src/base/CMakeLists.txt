
set(_srcs
  allocator.cc
  executable_cache.cc
  hlo_executor.cc
  hlo_executor.cu
  hlo_loader.cc
  hlo_loader.cu
  legate_mapper.cc
  registration.cc
  task_utils.cc
  xla_task.cc
)

add_library(
  legate_xla_base
  OBJECT
  ${_srcs}
)

target_include_directories(legate_xla_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIRECTORY})
target_link_libraries(legate_xla_base PUBLIC legate::core)

if (LegateXLA_HLO_RUNNER)
  target_compile_definitions(legate_xla_base PRIVATE LegateXLA_HLO_RUNNER)
endif()
