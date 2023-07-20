
include(${CMAKE_SOURCE_DIR}/cmake/third_party/get_xla.cmake)

set(_srcs client.cc)

add_library(legate_plugin ${_srcs})

target_link_libraries(legate_plugin PRIVATE xla::xla legate::xla_runtime)
target_include_directories(legate_plugin PRIVATE ${CMAKE_SOURCE_DIR}/src/base)

set_target_properties(legate_plugin PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)


set_target_properties(legate_plugin
    PROPERTIES INSTALL_RPATH "\$ORIGIN")

install(TARGETS legate_plugin
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
