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

#include "allocator.h"
#include "legate_xla_common.h"
#include <realm.h>
#include <realm/event.h>
#include <shape.h>
#include <tiled_array.h>
#include <zuku/defer.h>

namespace legate_xla {

TempBufferAllocator::TempBufferAllocator(const zuku::ArrayTile &array)
    : size_(array.byte_size()), base_ptr_(array.ptr<char>()), freed_size_(0) {
  next_ptr_ = base_ptr_;
}

void *TempBufferAllocator::Allocate(size_t size) {
  const int64_t size_to_allocate = AlignTempSize(size);

  void *ret_ptr = const_cast<char *>(next_ptr_);
  next_ptr_ += size_to_allocate;

  return ret_ptr;
}

void TempBufferAllocator::Free(void *buf, size_t size) {
  const int64_t size_allocated = AlignTempSize(size);
  freed_size_ += size_allocated;
  // we are a dumb allocator, only reset the next pointer
  // once all previous temp allocations have been freed
  if (freed_size_ == size_) {
    next_ptr_ = base_ptr_;
  }
}

void *DynamicBufferAllocator::Allocate(size_t size) {
  zuku::TileShape shape{
      .type = zuku::SupportedType::S8,
      .dims = {(int64_t)size},
  };
  auto [event, tile] =
      zuku::ArrayTile::Create(std::move(shape), {.processor = proc_});
  // we don't have a good way to be asynchronous with XLA
  event.wait();

  void *data = tile.data();
  tiles_.insert({tile.data(), std::move(tile)});
  return data;
}

void DynamicBufferAllocator::Free(void *buf, size_t size) { tiles_.erase(buf); }

} // namespace legate_xla
