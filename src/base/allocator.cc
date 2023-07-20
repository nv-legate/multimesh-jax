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

namespace legate_xla {

using namespace Legion;
using namespace legate;

DeferredBufferAllocator::DeferredBufferAllocator() {
  auto proc = Processor::get_executing_processor();
  mem_kind = proc.kind() == Processor::TOC_PROC ? Memory::GPU_FB_MEM
                                                : Memory::SYSTEM_MEM;
}

void *DeferredBufferAllocator::Allocate(size_t size) {
  Buffer buffer(mem_kind, legate::Rect<1>(0, size - 1));
  void *p = buffer.ptr(0);
#ifdef DEBUG_LEGATE_LLM
  assert(buffers.find(p) == buffers.end());
#endif
  buffers[p] = buffer;
  return p;
}

void DeferredBufferAllocator::Free(void *buf, size_t size) {
  auto finder = buffers.find(buf);
#ifdef DEBUG_LEGATE_LLM
  assert(finder != buffers.end());
#endif
  finder->second.destroy();
  buffers.erase(finder);
}

} // namespace legate_xla
