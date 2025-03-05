function(find_or_configure_xla)
  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(OpenXLA version git_repo git_branch shallow exclude_from_all)

  set(LegateJAX_BAZEL_REMOTE_CACHE "" CACHE STRING "An optional remote cache for bazel builds")

  if (xla_REPOSITORY)
    set(git_repo ${legate_core_REPOSITORY})
  endif()

  if (xla_BRANCH)
    set(git_branch ${legate_core_BRANCH})
  endif()

  rapids_cpm_find(xla ${version}
      GLOBAL_TARGETS     xla::xla
      BUILD_EXPORT_SET   legate-jax-exports
      INSTALL_EXPORT_SET legate-jax-exports
      CPM_ARGS
        GIT_REPOSITORY ${git_repo}
        GIT_TAG        ${git_tag}
        DOWNLOAD_ONLY
  )

  set(xla_client_library_name liblegate_xla_client.so)
  set(xla_compiler_library_name legate_xla_compiler.so)
  set(xla_client_library "${xla_SOURCE_DIR}/bazel-bin/xla/pjrt/legate/${xla_client_library_name}")
  set(xla_compiler_library "${xla_SOURCE_DIR}/bazel-bin/xla/pjrt/legate/${xla_compiler_library_name}")

  file(GLOB xla_source_files
       "${xla_SOURCE_DIR}/xla/pjrt/legate/*.cc"
       "${xla_SOURCE_DIR}/xla/pjrt/legate/*.h"
       "${xla_SOURCE_DIR}/xla/pjrt/legate/BUILD")


  set(test_names
    mpmd_partition_test
    legate_buffer_action_test
    legate_sharding_test
    loop_scheduler_test
    legate_pjrt_client_test
    legate_pjrt_executable_test
    mpmd_sharding_propagation_test
    mpmd_simple_loop_increment_coloring_test
    mpmd_coloring_test
    mpmd_cut_size_minimizer_test
    mpmd_computation_grouper_test
    mpmd_microbatch_loop_canonicalizer_test
    mpmd_cross_task_barrier_remover_test
    mpmd_logical_sharding_propagation_test
    mpmd_computation_fusion_test
    mpmd_logical_to_gspmd_sharding_test
    mpmd_instruction_delay_recolor_test
    mpmd_unused_param_output_remover_test
    mpmd_unused_loop_output_remover_test
    mpmd_argument_recompute_test
    mpmd_hoist_loop_convert_test
    mpmd_hoist_shard_map_reduce_test
    mpmd_shard_map_loop_reduce_test
    mpmd_utils_test
    mpmd_loop_unroll_test
    mpmd_insert_reshard_test
    mpmd_buffer_scheduling_name_test
  )

  set(target_names
    "//xla/pjrt/legate:liblegate_xla_client.so"
    "//xla/pjrt/legate:legate_xla_compiler.so"
  )
  if (LegateJAX_ENABLE_TESTS)
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
 if (LegateJAX_ASAN)
   list (APPEND _bazel_options
     --copt -fsanitize=address
     --linkopt -fsanitize=address)
 endif()

 if (LegateJAX_BAZEL_REMOTE_CACHE)
    list (APPEND _bazel_options
     --remote_cache ${LegateJAX_BAZEL_REMOTE_CACHE})
 endif()

 if (DEFINED zuku_SOURCE_DIR)
  list(APPEND _bazel_options
    --override_repository=zuku=${zuku_SOURCE_DIR})
  list(APPEND _bazel_options
    --override_repository=realm=${zuku_SOURCE_DIR}/realm)
 endif()

 add_custom_command(
    OUTPUT  ${xla_client_library}
    OUTPUT  ${xla_compiler_library}
    COMMENT "Building XLA components ${target_names}..."
    COMMAND rm -rf "${xla_SOURCE_DIR}/bazel-bin" && XLA_LEGATE_SOURCE_DIR=${CMAKE_SOURCE_DIR} bazel --batch build ${_bazel_options} ${target_names} --check_visibility=false
    WORKING_DIRECTORY ${xla_SOURCE_DIR}
    DEPENDS ${xla_source_files}
    USES_TERMINAL
    VERBATIM
  )

  add_custom_command(
      OUTPUT  ${PROJECT_BINARY_DIR}/lib/${xla_client_library_name}
      POST_BUILD
      DEPENDS ${xla_client_library}
      COMMAND ${CMAKE_COMMAND} -E copy ${xla_client_library} ${PROJECT_BINARY_DIR}/lib
      VERBATIM
  )

  add_custom_command(
      OUTPUT  ${PROJECT_BINARY_DIR}/lib/${xla_compiler_library_name}
      POST_BUILD
      DEPENDS ${xla_compiler_library}
      COMMAND ${CMAKE_COMMAND} -E copy ${xla_compiler_library} ${PROJECT_BINARY_DIR}/lib/libxla_compiler_plugin.so
      VERBATIM
  )

  add_library(xla SHARED IMPORTED GLOBAL)
  add_library(xla_compiler_plugin SHARED IMPORTED GLOBAL)

  add_custom_target(xla_build ALL
    DEPENDS ${xla_library} 
      ${PROJECT_BINARY_DIR}/lib/${xla_client_library_name}
      ${PROJECT_BINARY_DIR}/lib/${xla_compiler_library_name}
  )

  add_dependencies(xla xla_build)
  add_dependencies(xla_compiler_plugin xla_build)

  add_library(xla::xla ALIAS xla)
  add_library(xla::xla_compiler_plugin ALIAS xla_compiler_plugin)

  set_target_properties(xla
    PROPERTIES
      IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/${xla_client_library_name}
      IMPORTED_NO_SONAME TRUE
  )

  set_target_properties(xla_compiler_plugin
    PROPERTIES
      IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/lib/libxla_compiler_plugin.so
      IMPORTED_NO_SONAME TRUE
  )

  install(IMPORTED_RUNTIME_ARTIFACTS xla)

  include(GoogleTest)

  if (LegateJAX_ENABLE_TESTS)
    foreach(test_exe ${test_names})
      add_executable(${test_exe} IMPORTED)
      set_target_properties(${test_exe} PROPERTIES IMPORTED_LOCATION ${xla_SOURCE_DIR}/bazel-bin/xla/pjrt/legate/${test_exe})
      add_dependencies(${test_exe} xla_build)
      gtest_discover_tests(${test_exe}
         WORKING_DIRECTORY ${xla_SOURCE_DIR}
         DISCOVERY_TIMEOUT 10
         DISCOVERY_MODE PRE_TEST)
    endforeach()
  endif()


endfunction()

find_or_configure_xla()
