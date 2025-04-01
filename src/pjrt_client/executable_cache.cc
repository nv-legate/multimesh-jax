/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "executable_cache.h"

#include <iostream>
#include <mutex>

namespace xla {

static std::mutex cache_lock;

bool ExecutableCache::compile_executable(
    uint64_t hlo_id,
    std::function<std::unique_ptr<LegateExecutable>()> invoke) {
  cache_lock.lock();
  auto &entry = executables_[hlo_id];
  cache_lock.unlock();

  std::lock_guard<std::mutex> guard(entry.lock);
  if (entry.executable) {
    return false;
  }

  entry.executable = invoke();
  return true;
}

LegateExecutable *ExecutableCache::find_executable(uint64_t hlo_id) {
  std::lock_guard<std::mutex> guard(cache_lock);
  auto finder = executables_.find(hlo_id);
  if (executables_.end() == finder) {
    return nullptr;
  }
  return finder->second.executable.get();
}

static ExecutableCache &get_executable_cache() {
  static ExecutableCache executable_cache;
  return executable_cache;
}

bool compile_executable(
    uint64_t hlo_id,
    std::function<std::unique_ptr<LegateExecutable>()> invoke) {
  return get_executable_cache().compile_executable(hlo_id, std::move(invoke));
}

LegateExecutable *find_executable(uint64_t hlo_id) {
  return get_executable_cache().find_executable(hlo_id);
}

}  // namespace xla