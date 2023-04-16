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
#include "legate_xla.h"

#include <mutex>

namespace legate_xla {

static std::mutex cache_lock;

void ExecutableCache::register_executable(uint64_t hlo_id,
                                          std::unique_ptr<LegateExecutable> executable)
{
  std::lock_guard<std::mutex> guard(cache_lock);
  auto iter = executables_.find(hlo_id);
  // someone should have claimed the compile token for this previously
  assert(iter != executables_.end());
  iter->second = std::move(executable);
}

bool ExecutableCache::claim_executable_compile_token(uint64_t hlo_id)
{
  std::lock_guard<std::mutex> guard(cache_lock);
  if (executables_.find(hlo_id) != executables_.end()) { return false; }
  // drop an empty entry to keep others from building this
  executables_[hlo_id];
  return true;
}

LegateExecutable* ExecutableCache::find_executable(uint64_t hlo_id)
{
  std::lock_guard<std::mutex> guard(cache_lock);
  auto finder = executables_.find(hlo_id);
  if (executables_.end() == finder) {
    return nullptr;
  }
  return finder->second.get();
}

static ExecutableCache& get_executable_cache()
{
  static ExecutableCache executable_cache;
  return executable_cache;
}

void register_executable(uint64_t hlo_id,
                         std::unique_ptr<LegateExecutable> executable)
{
  get_executable_cache().register_executable(hlo_id,
                                             std::move(executable));
}

bool claim_executable_compile_token(uint64_t hlo_id)
{
  return get_executable_cache().claim_executable_compile_token(hlo_id);
}

LegateExecutable* find_executable(uint64_t hlo_id)
{
  return get_executable_cache().find_executable(hlo_id);
}


}  // namespace legate_xla