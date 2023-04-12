
add_library(
  legate_xla
  STATIC
  allocator.cc
  hlo_executor.cc
  hlo_executor.cu
  hlo_loader.cc
  hlo_loader.cu
  task_utils.cc
  xla_task.cc
)
set_target_properties(legate_xla PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

file(COPY legate_xla.h DESTINATION ${PROJECT_BINARY_DIR}/include)

target_include_directories(legate_xla PRIVATE ${CMAKE_CURRENT_SOURCE_DIRECTORY})
target_link_libraries(legate_xla PUBLIC legate::core)

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_xla
        DESTINATION ${lib_dir}
        EXPORT legate-xla-exports)
