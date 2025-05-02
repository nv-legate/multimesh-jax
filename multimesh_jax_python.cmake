#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

##############################################################################
# - User Options  ------------------------------------------------------------


set(libpath "")

set(pyroot jax_plugins/multimesh)
set(libname libmultimesh_plugin)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_info.py.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/${pyroot}/install_info.py"
@ONLY)

add_library(xla_python INTERFACE)
add_library(multimesh::xla_python ALIAS xla_python)
target_link_libraries(xla_python INTERFACE Legion::Realm Legion::RealmRuntime)

##############################################################################
# - install targets ----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS xla_python
        DESTINATION ${lib_dir}
        EXPORT multimesh-jax-python-exports)

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide Python targets for MultiMesh

Imported Targets:
  - multimesh::xla_python

]=])

rapids_export(
  INSTALL xla_python
  EXPORT_SET multimesh-jax-python-exports
  GLOBAL_TARGETS xla_python
  NAMESPACE multimesh::
  DOCUMENTATION doc_string)

# build export targets
rapids_export(
  BUILD xla_python
  EXPORT_SET multimesh-jax-python-exports
  GLOBAL_TARGETS xla_python
  NAMESPACE multimesh::
  DOCUMENTATION doc_string)
