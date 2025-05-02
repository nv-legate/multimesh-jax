/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "executable_cache.h"

#include <iostream>
#include <mutex>

namespace xla {

static std::mutex cache_lock;

bool ExecutableCache::compile_executable(
    uint64_t hlo_id,
    std::function<std::unique_ptr<MultiMeshExecutable>()> invoke) {
  cache_lock.lock();
  auto& entry = executables_[hlo_id];
  cache_lock.unlock();

  std::lock_guard<std::mutex> guard(entry.lock);
  if (entry.executable) {
    return false;
  }

  entry.executable = invoke();
  return true;
}

MultiMeshExecutable* ExecutableCache::find_executable(uint64_t hlo_id) {
  std::lock_guard<std::mutex> guard(cache_lock);
  auto finder = executables_.find(hlo_id);
  if (executables_.end() == finder) {
    return nullptr;
  }
  return finder->second.executable.get();
}

static ExecutableCache& get_executable_cache() {
  static ExecutableCache executable_cache;
  return executable_cache;
}

bool compile_executable(
    uint64_t hlo_id,
    std::function<std::unique_ptr<MultiMeshExecutable>()> invoke) {
  return get_executable_cache().compile_executable(hlo_id, std::move(invoke));
}

MultiMeshExecutable* find_executable(uint64_t hlo_id) {
  return get_executable_cache().find_executable(hlo_id);
}

}  // namespace xla
