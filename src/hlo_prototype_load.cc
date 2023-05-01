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

#include "hlo_prototype_load.h"
#include "allocator.h"
#include "executable_cache.h"
#include "hlo_loader.h"
#include "legate_to_xla.h"
#include "task_utils.h"

using namespace legate;

namespace legate_xla {

/*static*/ void
HLOPrototypeLoaderTask::load_and_compile(TaskContext &context,
                                         const std::string &platform_name) {
  auto &scalars = context.scalars();
  uint64_t run_id = scalars[0].value<uint64_t>();
  auto hlo_string = scalars[1].value<std::string>();
  auto hlo_name = scalars[2].value<std::string>();
  auto hlo_id = scalars[3].value<uint64_t>();
  auto loader_npartitions = scalars[4].value<uint32_t>();

  auto compiler = GetLegateCompilerFromHloProtoText(
      hlo_string, platform_name, /*replica_count=*/1, loader_npartitions);

  if (!compiler) {
    std::cerr << "Failed to get compiler" << std::endl;
    LEGATE_ABORT;
  }

  if (claim_executable_compile_token(hlo_id)) {
    HLOLoaderTask::load_and_compile(
        context, compiler.get(), run_id, platform_name, loader_npartitions,
        /*has_sync_store=*/false, /*print_stats=*/true);
    register_executable(hlo_id, compiler->MakeExecutable());
  }
}

/*static*/ void HLOPrototypeLoaderTask::cpu_variant(TaskContext &context) {
  load_and_compile(context, "cpu");
}

namespace // unnamed
{
static void __attribute__((constructor)) register_tasks(void) {
  HLOPrototypeLoaderTask::register_variants();
}
} // namespace

} // namespace legate_xla
