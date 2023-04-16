
add_library(
  legate_xla
  allocator.cc
  hlo_executor.cc
  hlo_executor.cu
  hlo_loader.cc
  hlo_loader.cu
  hlo_prototype_distributed_init.cc
  hlo_prototype_distributed_init.cu
  hlo_prototype_distributed_shutdown.cc
  hlo_prototype_distributed_shutdown.cu
  hlo_prototype_fill.cc
  hlo_prototype_fill.cu
  hlo_prototype_load.cc
  hlo_prototype_load.cu
  hlo_prototype_execute.cc
  hlo_prototype_execute.cu
  legate_mapper.cc
  legate_xla.cc
  task_utils.cc
  xla_task.cc
)

#  legate_runtime.cc

set_target_properties(legate_xla PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

file(COPY legate_xla.h DESTINATION ${PROJECT_BINARY_DIR}/include)

target_include_directories(legate_xla PRIVATE ${CMAKE_CURRENT_SOURCE_DIRECTORY})
target_link_libraries(legate_xla PUBLIC legate::core)

if (LegateXLA_ENABLE_PYTHON)
  target_compile_definitions(legate_xla PRIVATE LEGATE_XLA_PYTHON_PROTOTYPE)
endif()

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_xla
        DESTINATION ${lib_dir}
        EXPORT legate-xla-exports)
