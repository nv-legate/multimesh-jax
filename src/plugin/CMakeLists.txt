
include(${CMAKE_SOURCE_DIR}/cmake/third_party/get_xla.cmake)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

set(_srcs client.cc)

add_library(legate_plugin ${_srcs})

target_link_libraries(legate_plugin PRIVATE xla::xla legate::xla_runtime)
target_include_directories(legate_plugin PRIVATE ${CMAKE_SOURCE_DIR}/src/base)

set_target_properties(legate_plugin PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

pybind11_add_module(legate_xla_compiler SHARED dummy.cc)
target_link_libraries(legate_xla_compiler PRIVATE xla::xla_compiler_plugin)
set_target_properties(legate_xla_compiler PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/legate/jax
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/legate/jax)

pybind11_add_module(legate_jax_impl SHARED legate_jax_impl.cc)
set_target_properties(legate_jax_impl PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/legate/jax
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/legate/jax)

target_link_libraries(legate_jax_impl PRIVATE legate::xla_runtime xla::xla)
set_target_properties(legate_jax_impl PROPERTIES INSTALL_RPATH "$ORIGIN/../../../..:$ORIGIN/../../build/lib")
set_target_properties(legate_jax_impl PROPERTIES SKIP_BUILD_RPATH FALSE)

set_target_properties(legate_plugin PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

set_target_properties(legate_plugin
    PROPERTIES INSTALL_RPATH "\$ORIGIN")

install(TARGETS legate_plugin
        DESTINATION ${lib_dir}
        EXPORT legate-xla-exports)

rapids_export(
  INSTALL LegateJAX
  EXPORT_SET legate-xla-exports
  GLOBAL_TARGETS legate_plugin
  NAMESPACE legate::
  LANGUAGES ${ENABLED_LANGUAGES}
)

# build export targets
rapids_export(
  BUILD LegateJAX
  EXPORT_SET legate-xla-exports
  GLOBAL_TARGETS legate_plugin
  NAMESPACE legate::
  LANGUAGES ${ENABLED_LANGUAES}
)
