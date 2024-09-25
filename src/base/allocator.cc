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
#include <realm.h>

namespace legate_xla {

DeferredBufferAllocator::DeferredBufferAllocator() {
  auto proc = Realm::Processor::get_executing_processor();
  mem_kind = proc.kind() == Realm::Processor::TOC_PROC ? Realm::Memory::GPU_FB_MEM
                                                : Realm::Memory::SYSTEM_MEM;
}

void *DeferredBufferAllocator::Allocate(size_t size) {
  throw std::runtime_error("DeferredBufferAllocator::Allocate: unimplmeneted");
  return nullptr;
}

void DeferredBufferAllocator::Free(void *buf, size_t size) {
  throw std::runtime_error("DeferredBufferAllocator::Free: unimplmeneted");
}

} // namespace legate_xla
