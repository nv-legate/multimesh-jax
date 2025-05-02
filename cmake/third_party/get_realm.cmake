#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

function(find_or_configure_realm)

  set(PKG_VERSION ${MultiMeshJAX_VERSION})
  include("${rapids-cmake-dir}/export/detail/parse_version.cmake")
  rapids_export_parse_version(${PKG_VERSION} realm PKG_VERSION)

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(Legion version git_repo git_branch shallow exclude_from_all)

  set(version ${PKG_VERSION})

  if (realm_REPOSITORY)
    set(git_repo ${realm_REPOSITORY})
  endif()

  if (realm_BRANCH)
    set(git_branch ${realm_BRANCH})
  endif()

  set(FIND_PKG_ARGS
      GLOBAL_TARGETS Legion::Realm Legion::RealmRuntime
      BUILD_EXPORT_SET   multimesh-jax-exports
      INSTALL_EXPORT_SET multimesh-jax-exports)

  if (NOT DEFINED Legion_ROOT AND DEFINED realm_ROOT)
    set(Legion_ROOT ${realm_ROOT})
  endif()

  if((NOT CPM_realm_SOURCE) AND (NOT CPM_DOWNLOAD_realm))
    set(_find_mode QUIET)
    if(realm_DIR OR realm_ROOT)
      set(_find_mode REQUIRED)
    endif()
    rapids_find_package(Legion ${version} EXACT CONFIG ${_find_mode} ${FIND_PKG_ARGS})
  endif()

  if(Legion_FOUND)
    message(STATUS "CPM: using local package Realm@${version}")
  else()
    rapids_cpm_find(Legion ${version} ${FIND_PKG_ARGS}
        CPM_ARGS
          GIT_REPOSITORY ${git_repo}
          GIT_BRANCH ${git_branch}
          FIND_PACKAGE_ARGUMENTS EXACT
          OPTIONS
            Legion_USE_CUDA ON
    )
  endif()
endfunction()


find_or_configure_realm()
