#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

function(find_or_configure_zuku)

  set(PKG_VERSION ${LegateJAX_VERSION})
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${PKG_VERSION} zuku PKG_VERSION)

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(zuku version git_repo git_branch shallow exclude_from_all)

  set(version ${PKG_VERSION})

  if (zuku_REPOSITORY)
    set(git_repo ${zuku_REPOSITORY})
  endif()

  if (zuku_BRANCH)
    set(git_branch ${zuku_BRANCH})
  endif()

  set(FIND_PKG_ARGS
      GLOBAL_TARGETS zuku::zuku
      BUILD_EXPORT_SET   legate-jax-exports
      INSTALL_EXPORT_SET legate-jax-exports)

  if((NOT CPM_zuku_SOURCE) AND (NOT CPM_DOWNLOAD_zuku))
    set(_find_mode QUIET)
    if(zuku_DIR OR zuku_ROOT)
      set(_find_mode REQUIRED)
    endif()
    rapids_find_package(zuku EXACT CONFIG ${_find_mode} ${FIND_PKG_ARGS})
  endif()

  if(zuku_FOUND)
    message(STATUS "CPM: using local package Zuku")
  else()
    rapids_cpm_find(zuku ${PKG_VERSION} ${FIND_PKG_ARGS}
        CPM_ARGS
          GIT_REPOSITORY ${git_repo}
          GIT_BRANCH ${git_branch}
          FIND_PACKAGE_ARGUMENTS EXACT
          OPTIONS
    )
  endif()
endfunction()


find_or_configure_zuku()
