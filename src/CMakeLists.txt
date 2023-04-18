
set(_srcs
  allocator.cc
  executable_cache.cc
  hlo_executor.cc
  hlo_executor.cu
  hlo_loader.cc
  hlo_loader.cu
  legate_mapper.cc
  legate_xla.cc
  task_utils.cc
  xla_task.cc
)

if (LegateXLA_ENABLE_PYTHON)
  list(APPEND _srcs
    hlo_prototype_distributed_init.cc
    hlo_prototype_distributed_init.cu
    hlo_prototype_distributed_shutdown.cc
    hlo_prototype_distributed_shutdown.cu
    hlo_prototype_fill.cc
    hlo_prototype_fill.cu
    hlo_prototype_load.cc
    hlo_prototype_load.cu
    hlo_prototype_execute.cc
    hlo_prototype_execute.cu)
else()
  list(APPEND _srcs
    legate_runtime.cc)
endif()

add_library(
  legate_xla
  ${_srcs}
)

set_target_properties(legate_xla PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

file(COPY
  legate_xla_common.h
  legate_to_xla.h
  xla_to_legate.h
  DESTINATION ${PROJECT_BINARY_DIR}/include)

target_include_directories(legate_xla PRIVATE ${CMAKE_CURRENT_SOURCE_DIRECTORY})
target_link_libraries(legate_xla PUBLIC legate::core)

if (LegateXLA_ENABLE_PYTHON)
  target_compile_definitions(legate_xla PRIVATE LEGATE_XLA_PYTHON_PROTOTYPE)
endif()

if (LegateXLA_ENABLE_PYTHON)
  include(${CMAKE_SOURCE_DIR}/cmake/third_party/get_xla.cmake)
  target_link_libraries(legate_xla PRIVATE xla)
endif()

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_xla
        DESTINATION ${lib_dir}
        EXPORT legate-xla-exports)

rapids_export(
  INSTALL LegateXLA
  EXPORT_SET legate-xla-exports
  GLOBAL_TARGETS xla-bridge
  NAMESPACE legate::
  LANGUAGES ${ENABLED_LANGUAGES}
)

# build export targets
rapids_export(
  BUILD LegateXLA
  EXPORT_SET legate-xla-exports
  GLOBAL_TARGETS xla-birdge
  NAMESPACE legate::
  LANGUAGES ${ENABLED_LANGUAES}
)
