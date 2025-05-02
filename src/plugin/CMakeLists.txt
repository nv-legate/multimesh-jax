#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include(${CMAKE_SOURCE_DIR}/cmake/third_party/get_xla.cmake)
find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)

set(_srcs client.cc)

add_library(multimesh_plugin ${_srcs})

target_link_libraries(multimesh_plugin PUBLIC zuku::zuku xla::xla)

pybind11_add_module(multimesh_xla_compiler SHARED dummy.cc)
target_link_libraries(multimesh_xla_compiler PRIVATE xla::xla_compiler_plugin)
set_target_properties(multimesh_xla_compiler PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/multimesh/jax
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/multimesh/jax)

pybind11_add_module(multimesh_jax_impl SHARED multimesh_jax_impl.cc)
target_link_libraries(multimesh_jax_impl PRIVATE zuku::zuku)
set_target_properties(multimesh_jax_impl PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/multimesh/jax
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/multimesh/jax)

target_link_libraries(multimesh_jax_impl PRIVATE xla::xla)
#set_target_properties(multimesh_jax_impl PROPERTIES INSTALL_RPATH "$ORIGIN/../../../..:$ORIGIN/../../build/lib")
set_target_properties(multimesh_jax_impl PROPERTIES SKIP_BUILD_RPATH FALSE)

set_target_properties(multimesh_plugin PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
         ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set_target_properties(multimesh_plugin
    PROPERTIES INSTALL_RPATH "\$ORIGIN")

install(TARGETS multimesh_plugin
        DESTINATION ${lib_dir}
        EXPORT multimesh-jax-exports)

rapids_export(
  INSTALL MultiMeshJAX
  EXPORT_SET multimesh-jax-exports
  GLOBAL_TARGETS multimesh_plugin
  NAMESPACE multimesh::
  LANGUAGES ${ENABLED_LANGUAGES}
)

# build export targets
rapids_export(
  BUILD MultiMeshJAX
  EXPORT_SET multimesh-jax-exports
  GLOBAL_TARGETS multimesh_plugin
  NAMESPACE multimesh::
  LANGUAGES ${ENABLED_LANGUAES}
)
