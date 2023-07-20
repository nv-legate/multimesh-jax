

set(_srcs
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

add_library(
  legate_hlo_runner
  ${_srcs}
)

set_target_properties(legate_hlo_runner PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

target_include_directories(legate_hlo_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIRECTORY})
target_include_directories(legate_hlo_runner PRIVATE ${CMAKE_SOURCE_DIR}/src/base)
target_link_libraries(legate_hlo_runner PRIVATE legate_xla_base)

include(${CMAKE_SOURCE_DIR}/cmake/third_party/get_xla.cmake)

target_link_libraries(legate_hlo_runner PRIVATE xla)

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_hlo_runner
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
  GLOBAL_TARGETS xla-bridge
  NAMESPACE legate::
  LANGUAGES ${ENABLED_LANGUAES}
)
