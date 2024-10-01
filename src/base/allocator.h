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

#include "legate_xla_common.h"
#include <processor.h>
#include <realm.h>
#include <unordered_map>
#include <zuku/tiled_array.h>

namespace legate_xla {

class TempBufferAllocator : public TaskMemoryAllocator {
public:
  TempBufferAllocator(const zuku::ArrayTile &temp);
  void *Allocate(size_t size) override;
  void Free(void *buf, size_t size) override;

private:
  const char *base_ptr_;
  const char *next_ptr_;
  const int64_t size_;
  int64_t freed_size_;
};

class DynamicBufferAllocator : public TaskMemoryAllocator {
public:
  DynamicBufferAllocator(zuku::Processor p) : proc_(std::move(p)) {}
  void *Allocate(size_t size) override;
  void Free(void *buf, size_t size) override;

private:
  std::unordered_map<void *, zuku::ArrayTile> tiles_;
  zuku::Processor proc_;
};

} // namespace legate_xla
