
set(_srcs
  allocator.cc
  executable_cache.cc
  hlo_executor.cc
  hlo_executor.cu
  hlo_loader.cc
  hlo_loader.cu
  legate_mapper.cc
  registration.cc
  shard_assemble.cc
  shard_assemble.cu
  shard_getter.cc
  shard_getter.cu
  task_utils.cc
  xla_task.cc
)

add_library(
  legate_xla_base
  OBJECT
  ${_srcs}
)

target_include_directories(legate_xla_base PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(legate_xla_base PUBLIC Legion::Realm Legion::RealmRuntime)
target_link_libraries(legate_xla_base PUBLIC zuku::zuku)
add_library(legate::xla_base ALIAS legate_xla_base)

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS legate_xla_base
        DESTINATION ${lib_dir}
        EXPORT legate-xla-exports)
