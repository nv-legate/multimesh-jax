/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace xla {

class TaskMemoryAllocator {
 public:
  virtual void *Allocate(size_t size) = 0;

  virtual void Free(void *buf, size_t size) = 0;
};

static constexpr int64_t kTempMinAlignment = 4096;

inline int64_t AlignTempSize(int64_t size) {
  return ((size + kTempMinAlignment - 1) / kTempMinAlignment) *
         kTempMinAlignment;
}

}  // namespace xla
