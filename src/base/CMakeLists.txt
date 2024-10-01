
set(_srcs
  allocator.cc
  executable_cache.cc
  hlo_executor.cc
  hlo_loader.cc
  task_utils.cc
  xla_task.cc
)

add_library(
  legate_xla_base
  OBJECT
  ${_srcs}
)
set_target_properties(legate_xla_base PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

target_include_directories(legate_xla_base PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(legate_xla_base PUBLIC Legion::Realm Legion::RealmRuntime)
target_link_libraries(legate_xla_base PUBLIC zuku::zuku)
add_library(legate::xla_base ALIAS legate_xla_base)

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_xla_base
        DESTINATION ${lib_dir}
        EXPORT legate-jax-exports)
