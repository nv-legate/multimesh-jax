
if (LegateJAX_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
  add_compile_options(-fno-common)
endif()

set(_srcs
    legate_runtime.cc
    legate_xla.cc)

add_library(legate_xla_runtime ${_srcs})

set_target_properties(legate_xla_runtime PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

target_include_directories(legate_xla_runtime PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_link_libraries(legate_xla_runtime PUBLIC Legion::Realm Legion::RealmRuntime legate_xla_base zuku::zuku)

add_library(legate::xla_runtime ALIAS legate_xla_runtime)

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_xla_runtime
        DESTINATION ${lib_dir}
        EXPORT legate-xla-exports)