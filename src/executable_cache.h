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

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "legate_xla.h"

namespace legate_xla {


class ExecutableCache {
 public:
  void register_executable(uint64_t hlo_id,
                           std::unique_ptr<LegateExecutable> executable);

  bool claim_executable_compile_token(uint64_t hlo_id);
  LegateExecutable* find_executable(uint64_t hlo_id);
  //ExecutableInfo& get_executable_info(uint64_t hlo_id);

 private:
  std::unordered_map<uint64_t, std::unique_ptr<LegateExecutable>> executables_;
};

bool claim_executable_compile_token(uint64_t hlo_id);

void register_executable(uint64_t hlo_id,
                         std::unique_ptr<LegateExecutable> executable);

LegateExecutable* find_executable(uint64_t hlo_id);

}  // namespace llm
