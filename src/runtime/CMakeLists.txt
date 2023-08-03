
set(_srcs
    legate_runtime.cc
    legate_xla.cc)

add_library(legate_xla_runtime ${_srcs})

set_target_properties(legate_xla_runtime PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

target_include_directories(legate_xla_runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIRECTORY})
target_include_directories(legate_xla_runtime PRIVATE ${CMAKE_SOURCE_DIR}/src/base)
target_link_libraries(legate_xla_runtime PUBLIC legate::core)
target_link_libraries(legate_xla_runtime PRIVATE legate_xla_base)

add_library(legate::xla_runtime ALIAS legate_xla_runtime)


include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_xla_runtime
        DESTINATION ${lib_dir}
        EXPORT legate-xla-exports)
