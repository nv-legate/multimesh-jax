#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)
project(LegateJAX LANGUAGES C CXX CUDA)
set(LegateJAX_VERSION 24.9.0)

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

if (LegateJAX_RAPIDS_DIR)
  include(${LegateJAX_RAPIDS_DIR}/RAPIDS.cmake)
else()
  if(NOT rapids-cmake-version)
    # default
    set(rapids-cmake-version 25.06)
    set(rapids-cmake-sha "365322aca32fd6ecd7027f5d7ec7be50b7f3cc2a")
  endif()
  if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
    file(DOWNLOAD
         https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake
         ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  endif()
  include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
endif()
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

rapids_cpm_init(OVERRIDE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/versions.json)


enable_testing()

option(LegateJAX_ENABLE_CUDA ON "Whether to build with CUDA")
option(LegateJAX_ASAN OFF "Whether to build with address saniziter")
option(LegateJAX_ENABLE_TESTS OFF "Whether to build tests")

if (SKBUILD)
  find_package(LegateJAX)
else()
  set(LegateJAX_FOUND FALSE)
endif()

if (NOT LegateJAX_FOUND)
  include(cmake/third_party/get_realm.cmake)
  include(cmake/third_party/get_zuku.cmake)
  add_subdirectory(src)
endif()

include(GNUInstallDirs)
rapids_cmake_install_lib_dir(lib_dir)

if (SKBUILD)
  include(legate_xla_python.cmake)
endif()

