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

function(find_or_configure_legate_core)

  set(PKG_VERSION ${LegateXLA_VERSION})
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${PKG_VERSION} legate_core PKG_VERSION)

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(LegateCore version git_repo git_branch shallow exclude_from_all)

  set(version ${PKG_VERSION})

  if (legate_core_REPOSITORY)
    set(git_repo ${legate_core_REPOSITORY})
  endif()

  if (legate_core_BRANCH)
    set(git_branch ${legate_core_BRANCH})
  endif()

  set(FIND_PKG_ARGS
      GLOBAL_TARGETS     legate::core
      BUILD_EXPORT_SET   legate_xla-exports
      INSTALL_EXPORT_SET legate_xla-exports)

  if((NOT CPM_legate_core_SOURCE) AND (NOT CPM_DOWNLOAD_legate_core))
    set(_find_mode QUIET)
    if(legate_core_DIR OR legate_core_ROOT)
      set(_find_mode REQUIRED)
    endif()
    rapids_find_package(legate_core ${version} EXACT CONFIG ${_find_mode} ${FIND_PKG_ARGS})
  endif()

  if(legate_core_FOUND)
    message(STATUS "CPM: using local package Legate@${version}")
  else()
    rapids_cpm_find(legate_core ${version} ${FIND_PKG_ARGS}
        CPM_ARGS
          GIT_REPOSITORY ${git_repo}
          GIT_BRANCH ${git_branch}
          FIND_PACKAGE_ARGUMENTS EXACT
          OPTIONS
            Legion_USE_CUDA ON
    )
  endif()
endfunction()


find_or_configure_legate_core()
