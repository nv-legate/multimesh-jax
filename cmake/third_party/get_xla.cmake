#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

function(find_or_configure_xla)
  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(OpenXLA version git_repo git_branch shallow exclude_from_all)

  set(MultiMeshJAX_BAZEL_REMOTE_CACHE "" CACHE STRING "An optional remote cache for bazel builds")
  set(MultiMeshJAX_XLA_LINKER "" CACHE STRING "The linker to override the default linker chosen by Bazel")


  if (xla_REPOSITORY)
    set(git_repo ${xla_REPOSITORY})
  endif()

  if (xla_BRANCH)
    set(git_branch ${xla_BRANCH})
  endif()

  rapids_cpm_find(xla ${version}
      GLOBAL_TARGETS     xla::xla
      BUILD_EXPORT_SET   multimesh-jax-exports
      INSTALL_EXPORT_SET multimesh-jax-exports
      CPM_ARGS
        GIT_REPOSITORY ${git_repo}
        GIT_TAG        ${git_tag}
        DOWNLOAD_ONLY
  )

  set(xla_client_library_name libmultimesh_xla_client.so)
  set(xla_compiler_library_name multimesh_xla_compiler.so)
  set(xla_client_library "${xla_SOURCE_DIR}/bazel-bin/xla/pjrt/multimesh/${xla_client_library_name}")
  set(xla_compiler_library "${xla_SOURCE_DIR}/bazel-bin/xla/pjrt/multimesh/${xla_compiler_library_name}")

  file(GLOB xla_source_files
       "${PROJECT_SOURCE_DIR}/src/pjrt_client/*.cc"
       "${PROJECT_SOURCE_DIR}/src/pjrt_client/*.h"
       "${PROJECT_SOURCE_DIR}/src/pjrt_client/BUILD")

  set(xla_symlink_files)
  foreach(PATH ${xla_source_files})
    get_filename_component(FILE_NAME ${PATH} NAME)
    list(APPEND xla_symlink_files "${xla_SOURCE_DIR}/xla/pjrt/multimesh/${FILE_NAME}")
  endforeach()

  add_custom_command(
      OUTPUT ${xla_symlink_files}
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              ${PROJECT_SOURCE_DIR}/src/pjrt_client
              ${xla_SOURCE_DIR}/xla/pjrt/multimesh
      COMMENT "Symlink plugin client code into XLA source tree at ${xla_SOURCE_DIR}"
  )

  set(test_names
    mpmd_partition_test
    mm_buffer_action_test
    mm_sharding_test
    loop_scheduler_test
    mm_pjrt_client_test
    mm_pjrt_executable_test
    mpmd_sharding_propagation_test
    mpmd_simple_loop_increment_coloring_test
    mpmd_unpack_optimization_barrier_test
    mpmd_repack_optimization_barrier_test
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
    mpmd_reorder_shard_map_transpose_test
    mpmd_hoist_shard_map_reduce_test
    mpmd_shard_map_loop_reduce_test
    mpmd_utils_test
    mpmd_loop_unroll_test
    mpmd_insert_reshard_test
    mpmd_buffer_scheduling_name_test
  )

  set(target_names
    "//xla/pjrt/multimesh:libmultimesh_xla_client.so"
    "//xla/pjrt/multimesh:multimesh_xla_compiler.so"
  )
  if (MultiMeshJAX_ENABLE_TESTS)
    foreach(test ${test_names})
      list(APPEND target_names "//xla/pjrt/multimesh:${test}")
      list(APPEND xla_source_files "${xla_SOURCE_DIR}/xla/pjrt/multimesh/${test}.cc")
    endforeach()
  endif()

 set(_bazel_options
   --define open_source_build=true
   --define framework_shared_object=false
   --define tsl_protobuf_header_only=false
   --config=cuda
 )

 if (MultiMeshJAX_XLA_LINKER)
   list(APPEND _bazel_options 
     --linkopt=-fuse-ld=${MultiMeshJAX_XLA_LINKER})
   if (${MultiMeshJAX_XLA_LINKER} STREQUAL "lld")
     list(APPEND _bazel_options
       --linkopt -Wl,--undefined-version)
   endif()
 endif()
 
 if (MultiMeshJAX_ASAN)
   list (APPEND _bazel_options
     --copt -fsanitize=address
     --linkopt -fsanitize=address)
 endif()

 if (MultiMeshJAX_BAZEL_REMOTE_CACHE)
    list (APPEND _bazel_options
     --remote_cache ${MultiMeshJAX_BAZEL_REMOTE_CACHE})
 endif()

 if (DEFINED zuku_SOURCE_DIR)
  list(APPEND _bazel_options
    --override_repository=zuku=${zuku_SOURCE_DIR})
  list(APPEND _bazel_options
    --override_repository=realm=${zuku_SOURCE_DIR}/realm)
 endif()

 set(_bazel_startup_options 
 #  --batch
 )
 if (MultiMeshJAX_BAZEL_OUTPUT_BASE)
   list(APPEND _bazel_startup_options
     --output_base=${MultiMeshJAX_BAZEL_OUTPUT_BASE})
 endif()

 add_library(xla SHARED IMPORTED GLOBAL)
 add_library(xla_compiler_plugin SHARED IMPORTED GLOBAL)

 option(MultiMeshJAX_BUILD_XLA ON)

 if (CMAKE_LIBRARY_OUTPUT_DIRECTORY)
   set(LIB_FOLDER ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
 else()
   set(LIB_FOLDER ${PROJECT_BINARY_DIR}/lib)
 endif()

 add_custom_command(
   OUTPUT  ${xla_client_library}
   OUTPUT  ${xla_compiler_library}
   COMMENT "Building XLA components ${target_names}..."
   COMMAND rm -rf "${xla_SOURCE_DIR}/bazel-bin" && bazel ${_bazel_startup_option} build ${_bazel_options} ${target_names} --check_visibility=false
   WORKING_DIRECTORY ${xla_SOURCE_DIR}
   DEPENDS ${xla_symlink_files}
   USES_TERMINAL
   VERBATIM
 )

 add_custom_command(
     OUTPUT  ${LIB_FOLDER}/${xla_client_library_name}
     DEPENDS ${xla_client_library}
     COMMAND ${CMAKE_COMMAND} -E copy ${xla_client_library} ${LIB_FOLDER}
     VERBATIM
 )

 add_custom_command(
     OUTPUT  ${LIB_FOLDER}/${xla_compiler_library_name}
     DEPENDS ${xla_compiler_library}
     COMMAND ${CMAKE_COMMAND} -E copy ${xla_compiler_library} ${LIB_FOLDER}/libxla_compiler_plugin.so
     VERBATIM
 )

  file(GLOB xla_source_files
       "${PROJECT_SOURCE_DIR}/src/pjrt_client/*.cc"
       "${PROJECT_SOURCE_DIR}/src/pjrt_client/*.h"
       "${PROJECT_SOURCE_DIR}/src/pjrt_client/BUILD")


  set(xla_symlink_files)
  foreach(PATH ${xla_source_files})
    get_filename_component(FILE_NAME ${PATH} NAME)
    list(APPEND xla_symlink_files "${xla_SOURCE_DIR}/xla/pjrt/multimesh/${FILE_NAME}")
  endforeach()

  add_custom_target(xla_build ALL
    DEPENDS ${xla_library} 
      ${LIB_FOLDER}/${xla_client_library_name}
      ${LIB_FOLDER}/${xla_compiler_library_name}
  )

  add_dependencies(xla xla_build)
  add_dependencies(xla_compiler_plugin xla_build)

  add_library(xla::xla ALIAS xla)
  add_library(xla::xla_compiler_plugin ALIAS xla_compiler_plugin)

  set_target_properties(xla
    PROPERTIES
      IMPORTED_LOCATION ${LIB_FOLDER}/${xla_client_library_name}
      IMPORTED_NO_SONAME TRUE
  )

  set_target_properties(xla_compiler_plugin
    PROPERTIES
      IMPORTED_LOCATION ${LIB_FOLDER}/libxla_compiler_plugin.so
      IMPORTED_NO_SONAME TRUE
  )

  install(IMPORTED_RUNTIME_ARTIFACTS xla)
  install(IMPORTED_RUNTIME_ARTIFACTS xla_compiler_plugin)

  include(GoogleTest)

  if (MultiMeshJAX_ENABLE_TESTS)
    foreach(test_exe ${test_names})
      add_executable(${test_exe} IMPORTED)
      set_target_properties(${test_exe} PROPERTIES IMPORTED_LOCATION ${xla_SOURCE_DIR}/bazel-bin/xla/pjrt/multimesh/${test_exe})
      add_dependencies(${test_exe} xla_build)
      gtest_discover_tests(${test_exe}
         WORKING_DIRECTORY ${xla_SOURCE_DIR}
         DISCOVERY_TIMEOUT 10
         DISCOVERY_MODE PRE_TEST)
    endforeach()
  endif()


endfunction()

find_or_configure_xla()
