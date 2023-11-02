function(find_or_configure_xla)
  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(OpenXLA version git_repo git_branch shallow exclude_from_all)

  if (xla_REPOSITORY)
    set(git_repo ${legate_core_REPOSITORY})
  endif()

  if (xla_BRANCH)
    set(git_branch ${legate_core_BRANCH})
  endif()

  rapids_cpm_find(xla ${version}
      GLOBAL_TARGETS     xla::xla
      BUILD_EXPORT_SET   legate-xla-exports
      INSTALL_EXPORT_SET legate-xla-exports
      CPM_ARGS
        GIT_REPOSITORY ${git_repo}
        GIT_TAG        ${git_tag}
        DOWNLOAD_ONLY
  )

  if (LegateXLA_HLO_RUNNER)
    set(xla_library_name liblegate_hlo_prototype.so)
  else()
    set(xla_library_name liblegate_xla_client.so)
  endif()

  set(xla_library "${xla_SOURCE_DIR}/bazel-bin/xla/pjrt/legate/${xla_library_name}")

  file(GLOB xla_source_files
       "${xla_SOURCE_DIR}/xla/pjrt/legate/*.cc"
       "${xla_SOURCE_DIR}/xla/pjrt/legate/*.h"
       "${xla_SOURCE_DIR}/xla/pjrt/legate/BUILD")


  set(test_names
    hlo_partition_test
    legate_buffer_action_test
  )


  set(target_names "//xla/pjrt/legate:${xla_library_name}")
  if (LegateXla_ENABLE_TESTS)
    foreach(test ${test_names})
      list(APPEND target_names "//xla/pjrt/legate:${test}")
      list(APPEND xla_source_files "${xla_SOURCE_DIR}/xla/pjrt/legate/${test}.cc")
    endforeach()
  endif()

 set(_bazel_options
   --define open_source_build=true
   --define framework_shared_object=false
   --config=cuda
 )
 if (LegateXLA_ASAN)
   list (APPEND _bazel_options
     --copt -fsanitize=address
     --linkopt -fsanitize=address)
 endif()

 add_custom_command(
    OUTPUT  ${xla_library}
    COMMENT "Building XLA components..."
    COMMAND rm -rf "${xla_SOURCE_DIR}/bazel-bin" && XLA_LEGATE_SOURCE_DIR=${CMAKE_SOURCE_DIR} bazel --batch build ${_bazel_options} ${target_names} --check_visibility=false
    WORKING_DIRECTORY ${xla_SOURCE_DIR}
    DEPENDS ${xla_source_files}
    USES_TERMINAL
    VERBATIM
  )

  add_custom_command(
      OUTPUT  ${PROJECT_BINARY_DIR}/lib/${xla_library_name}
      POST_BUILD
      DEPENDS ${xla_library}
      COMMAND ${CMAKE_COMMAND} -E copy ${xla_library} ${PROJECT_BINARY_DIR}/lib
      VERBATIM
  )

  add_library(xla SHARED IMPORTED GLOBAL)

  add_custom_target(xla_build ALL
    DEPENDS ${xla_library} ${PROJECT_BINARY_DIR}/lib/${xla_library_name}
  )

  add_dependencies(xla xla_build)

  add_library(xla::xla ALIAS xla)

  set_target_properties(xla
    PROPERTIES
      IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/${xla_library_name}
      IMPORTED_NO_SONAME TRUE
  )

  install(IMPORTED_RUNTIME_ARTIFACTS xla)

  if (LegateXla_ENABLE_TESTS)
    foreach(test ${test_names})
      add_executable(${test} IMPORTED)
      set_target_properties(${test} PROPERTIES IMPORTED_LOCATION ${xla_SOURCE_DIR}/bazel-bin/xla/pjrt/legate/${test})
      add_dependencies(${test} xla_build)
      add_test(NAME ${test} COMMAND ${test} WORKING_DIRECTORY ${xla_SOURCE_DIR})
    endforeach()
  endif()


endfunction()

find_or_configure_xla()
