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
#include "legate_to_xla.h"
#include "task_utils.h"
#include <processor.h>

namespace legate_xla {

void LoadAndCompile(int64_t run_id, zuku::Processor p,
                    const std::shared_ptr<LegateCompiler> &compiler) {
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

} // namespace legate_xla
