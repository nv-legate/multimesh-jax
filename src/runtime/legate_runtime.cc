/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "legate_runtime.h"
#include "legate.h"
#include "legate_mapper.h"

using namespace Legion;
using namespace legate;

namespace legate_xla {

/*static*/ Runtime *Runtime::runtime_ = nullptr;

/*static*/ bool Runtime::synchronous_mode_ = false;

/*static*/ std::mutex Runtime::mutex_ = std::mutex();

Runtime::Runtime(legate::Runtime *core_runtime, legate::Library context)
    : core_runtime_(core_runtime), context_(context) {}

std::vector<LogicalStore> &Runtime::get_tmp_stores() {
  return temporary_stores;
}

AutoTask Runtime::create_task(XlaOpCode task_id) {
  return core_runtime_->create_task(context_, task_id);
}

ManualTask Runtime::create_task(XlaOpCode task_id,
                                const legate::Shape &launch_shape) {
  return core_runtime_->create_task(context_, task_id, launch_shape.extents());
}

void Runtime::submit(legate::AutoTask task) {
  core_runtime_->submit(std::move(task));
}

void Runtime::submit(legate::ManualTask task) {
  core_runtime_->submit(std::move(task));
}

void Runtime::issue_execution_fence(bool block) {
  core_runtime_->issue_execution_fence(block);
}

/*static*/ Runtime *Runtime::get_runtime() { return runtime_; }

/*static*/ bool Runtime::synchronous_mode() { return synchronous_mode_; }

/*static*/ std::mutex &Runtime::get_mutex() { return mutex_; }

/*static*/ void Runtime::initialize(legate::Runtime *core_runtime,
                                    legate::Library library) {
  if (nullptr == runtime_)
    runtime_ = new Runtime(core_runtime, library);

  char *env_str = getenv("LEGATE_XLA_SYNCHRONOUS_MODE");
  if (env_str && std::atoi(env_str) > 0) {
    log_xla.warning()
        << "Task synchronization is enabled via 'LEGATE_XLA_SYNCHRONOUS_MODE'. "
           "This should not be a production run.";
    synchronous_mode_ = true;
  }
}

} // namespace legate_xla