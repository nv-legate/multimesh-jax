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

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(LegateXLA LANGUAGES C CXX CUDA)
set(LegateXLA_VERSION 24.01.00)

set(CMAKE_CXX_STANDARD 17)

# This must be built as a shared library for
# the XLA extension. The XLA extension will privatize all
# symbols, which means the linked Legate runtime will
# be completely independent of all other Legate runtimes.
set(CMAKE_LINK_WHAT_YOU_USE ON)
set(BUILD_SHARED_LIBS ON)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
endif()

if(POLICY CMP0096)
  cmake_policy(SET CMP0096 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0096 NEW)
endif()

if(POLICY CMP0135)
  # make the timestamps of ExternalProject_ADD match the download time
  # https://cmake.org/cmake/help/latest/policy/CMP0135.html
  cmake_policy(SET CMP0135 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

##############################################################################
# - Download and initialize RAPIDS CMake helpers -----------------------------


if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.08/RAPIDS.cmake
       ${CMAKE_BINARY_DIR}/legate-xla-RAPIDS.cmake)
endif()
include(${CMAKE_BINARY_DIR}/legate-xla-RAPIDS.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/versions.json)

enable_testing()

option(LegateXLA_HLO_RUNNER OFF
       "Whether to build the HLO runner instead of the plugin client")
option(LegateXLA_ENABLE_CUDA ON "Whether to build with CUDA")

option(LegateXLA_ASAN OFF "Whether to build with address saniziter")

option(LegateXLA_ENABLE_TESTS "Whether to build the tests" ON)

if (SKBUILD)
  find_package(LegateXLA)
else()
  set(LegateXLA_FOUND FALSE)
endif()


if (NOT LegateXLA_FOUND)
  include(cmake/third_party/get_legate_core.cmake)
  add_subdirectory(src)
endif()

include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

if (SKBUILD)
  include(legate_xla_python.cmake)
elseif(NOT LegateXLA_HLO_RUNNER AND LegateXLA_ENABLE_TESTS)
  add_subdirectory(tests/cpp)
endif()

