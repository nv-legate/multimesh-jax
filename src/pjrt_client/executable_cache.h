/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "xla/pjrt/multimesh/mm_computation.h"

namespace xla {

class ExecutableCache {
  struct Entry {
    std::unique_ptr<MultiMeshExecutable> executable;
    std::mutex lock;
  };

 public:
  void register_executable(uint64_t hlo_id,
                           std::unique_ptr<MultiMeshExecutable> executable);

  bool compile_executable(
      uint64_t hlo_id,
      std::function<std::unique_ptr<MultiMeshExecutable>()> invoke);
  MultiMeshExecutable* find_executable(uint64_t hlo_id);

 private:
  std::unordered_map<uint64_t, Entry> executables_;
};

bool compile_executable(
    uint64_t hlo_id,
    std::function<std::unique_ptr<MultiMeshExecutable>()> invoke);

MultiMeshExecutable* find_executable(uint64_t hlo_id);

}  // namespace xla
