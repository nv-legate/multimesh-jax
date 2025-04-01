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
