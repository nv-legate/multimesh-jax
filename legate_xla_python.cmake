#=============================================================================
# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

##############################################################################
# - User Options  ------------------------------------------------------------


execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -E -DLEGATE_USE_PYTHON_CFFI
    -I "${CMAKE_CURRENT_SOURCE_DIR}/src/base"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/src/base/legate_xla_c.h"
  ECHO_ERROR_VARIABLE
  OUTPUT_VARIABLE header
  COMMAND_ERROR_IS_FATAL ANY
)

set(libpath "")

if (LegateXLA_HLO_RUNNER)
  set(pyroot lllm)
  set(libname liblegate_hlo_runner)
else()
  set(pyroot jax_plugins/legate)
  set(libname liblegate_plugin)
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_info.py.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/${pyroot}/install_info.py"
@ONLY)

add_library(xla_python INTERFACE)
add_library(legate::xla_python ALIAS xla_python)
target_link_libraries(xla_python INTERFACE legate::core)

##############################################################################
# - install targets ----------------------------------------------------------

include(CPack)
include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

install(TARGETS xla_python
        DESTINATION ${lib_dir}
        EXPORT legate-xla-python-exports)

##############################################################################
# - install export -----------------------------------------------------------

set(doc_string
        [=[
Provide Python targets for Legate LLM

Imported Targets:
  - legate::xla_python

]=])

set(code_string "")

rapids_export(
  INSTALL xla_python
  EXPORT_SET legate-xla-python-exports
  GLOBAL_TARGETS xla_python
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)

# build export targets
rapids_export(
  BUILD xla_python
  EXPORT_SET legate-xla-python-exports
  GLOBAL_TARGETS xla_python
  NAMESPACE legate::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string)
