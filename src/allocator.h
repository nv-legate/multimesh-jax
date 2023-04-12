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

#include "legate_xla.h"
#include "legate.h"
#include <unordered_map>

namespace legate_xla {

struct DeferredBufferAllocator : legate::TaskMemoryAllocator {
  using Buffer = Legion::DeferredBuffer<uint8_t, 1>;
  DeferredBufferAllocator();
  virtual void* Allocate(size_t size) override;
  virtual void Free(void* buf, size_t size) override;
  Legion::Memory::Kind mem_kind;
  std::unordered_map<void*, Buffer> buffers;
};

}  // namespace llm
