
include(${CMAKE_SOURCE_DIR}/cmake/third_party/get_xla.cmake)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

set(_srcs client.cc)

add_library(legate_plugin ${_srcs})

target_link_libraries(legate_plugin PUBLIC zuku::zuku xla::xla)

pybind11_add_module(legate_xla_compiler SHARED dummy.cc)
target_link_libraries(legate_xla_compiler PRIVATE xla::xla_compiler_plugin)
set_target_properties(legate_xla_compiler PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/legate/jax
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/legate/jax)

pybind11_add_module(legate_jax_impl SHARED legate_jax_impl.cc)
target_link_libraries(legate_jax_impl PRIVATE zuku::zuku)
set_target_properties(legate_jax_impl PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/legate/jax
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/legate/jax)

target_link_libraries(legate_jax_impl PRIVATE xla::xla)
#set_target_properties(legate_jax_impl PROPERTIES INSTALL_RPATH "$ORIGIN/../../../..:$ORIGIN/../../build/lib")
set_target_properties(legate_jax_impl PROPERTIES SKIP_BUILD_RPATH FALSE)

set_target_properties(legate_plugin PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set_target_properties(legate_plugin
    PROPERTIES INSTALL_RPATH "\$ORIGIN")

install(TARGETS legate_plugin
        DESTINATION ${lib_dir}
        EXPORT legate-jax-exports)

rapids_export(
  INSTALL LegateJAX
  EXPORT_SET legate-jax-exports
  GLOBAL_TARGETS legate_plugin
  NAMESPACE legate::
  LANGUAGES ${ENABLED_LANGUAGES}
)

# build export targets
rapids_export(
  BUILD LegateJAX
  EXPORT_SET legate-jax-exports
  GLOBAL_TARGETS legate_plugin
  NAMESPACE legate::
  LANGUAGES ${ENABLED_LANGUAES}
)
