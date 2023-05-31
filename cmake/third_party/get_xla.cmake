

rapids_cpm_find(xla 2.10
    GLOBAL_TARGETS     xla::xla
    BUILD_EXPORT_SET   legate-xla-exports
    INSTALL_EXPORT_SET legate-xla-exports
    CPM_ARGS
      GIT_REPOSITORY ssh://git@gitlab-master.nvidia.com:12051/legate/legate-xla-bridge.git
      GIT_TAG        main
      DOWNLOAD_ONLY
)

set(xla_library_name liblegate_hlo_prototype.so)
set(xla_library "${xla_SOURCE_DIR}/bazel-bin/xla/pjrt/legate/${xla_library_name}")
set(xla_target "//xla/pjrt/legate:${xla_library_name}")

file(GLOB xla_source_files
     "${xla_SOURCE_DIR}/xla/pjrt/legate/*.cc"
     "${xla_SOURCE_DIR}/xla/pjrt/legate/*.h"
     "${xla_SOURCE_DIR}/xla/pjrt/legate/BUILD")

set(_bazel_options
  --define open_source_build=true
  --define framework_shared_object=false
  --config=cuda)

add_custom_command(
  OUTPUT  ${xla_library}
  COMMENT "Building tensorflow components"
  COMMAND bazel --batch build ${_bazel_options} ${xla_target} //xla/tools:run_hlo_module --check_visibility=false
  COMMAND ${soname_command}
  WORKING_DIRECTORY ${xla_SOURCE_DIR}
  DEPENDS ${xla_source_files}
  USES_TERMINAL
  VERBATIM
)

add_library(xla SHARED IMPORTED GLOBAL)

add_custom_target(xla_build ALL
  DEPENDS ${xla_library}
)

add_dependencies(xla xla_build)

add_library(xla::xla ALIAS xla)

set_target_properties(xla
  PROPERTIES
    IMPORTED_LOCATION ${xla_library}
)
