/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "legate_computation.h"

namespace xla {

class ExecutableCache {
  struct Entry {
    std::unique_ptr<LegateExecutable> executable;
    std::mutex lock;
  };

 public:
  void register_executable(uint64_t hlo_id,
                           std::unique_ptr<LegateExecutable> executable);

  bool compile_executable(
      uint64_t hlo_id,
      std::function<std::unique_ptr<LegateExecutable>()> invoke);
  LegateExecutable *find_executable(uint64_t hlo_id);

 private:
  std::unordered_map<uint64_t, Entry> executables_;
};

bool compile_executable(
    uint64_t hlo_id, std::function<std::unique_ptr<LegateExecutable>()> invoke);

LegateExecutable *find_executable(uint64_t hlo_id);

}  // namespace xla
