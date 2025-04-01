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

#include "hlo_loader.h"

#include "allocator.h"
#include "executable_cache.h"
#include "src/zuku/processor.h"
#include "src/zuku/tiled_array.h"

namespace xla {
namespace {

class DynamicBufferAllocator : public TaskMemoryAllocator {
 public:
  explicit DynamicBufferAllocator(zuku::Processor p) : proc_(std::move(p)) {}

  void *Allocate(size_t size) override;
  void Free(void *buf, size_t size) override;

 private:
  std::unordered_map<void *, zuku::ArrayTile> tiles_;
  zuku::Processor proc_;
};

void *DynamicBufferAllocator::Allocate(size_t size) {
  zuku::TileShape shape{
      .type = zuku::SupportedType::S8,
      .dims = {(int64_t)size},
  };
  // log_xla.debug() << "Allocating temp of size " << size;
  auto [event, tile] =
      zuku::ArrayTile::Create(std::move(shape), {.processor = proc_});
  // we don't have a good way to be asynchronous with XLA
  event.wait();

  void *data = tile.data();
  tiles_.insert({tile.data(), std::move(tile)});
  return data;
}

void DynamicBufferAllocator::Free(void *buf, size_t size) { tiles_.erase(buf); }

}  // namespace

void LoadAndCompile(int64_t run_id, zuku::Processor p,
                    const std::shared_ptr<LegateCompiler> &compiler) {
  // log_xla.debug() << "Compiling " << compiler->Name() << " on "
  //                << p.global_id();
  // only one GPU per node should be running the compilation
  compile_executable(compiler->HloId(), [&] {
    DynamicBufferAllocator allocator{p};
    try {
      compiler->Compile(run_id, {.run_hlo_passes = true,
                                 .stream_executor_index = (int)p.local_id(),
                                 .allocator = &allocator,
                                 .print_stats = false});
    } catch (const std::exception &e) {
    } catch (...) {
    }
    return compiler->MakeExecutable();
  });
}

}  // namespace xla
