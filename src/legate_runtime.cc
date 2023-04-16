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

/*static*/ Runtime* Runtime::runtime_ = nullptr;

Runtime::Runtime(legate::Runtime* core_runtime,
                 legate::LibraryContext* context)
    : core_runtime_(core_runtime), context_(context) {}

std::vector<LogicalStore>& Runtime::get_tmp_stores() {
  return temporary_stores;
}

std::unique_ptr<AutoTask> Runtime::create_task(
    XlaOpCode task_id) {
  return core_runtime_->create_task(context_, task_id);
}

std::unique_ptr<ManualTask> Runtime::create_task(
    XlaOpCode task_id, const legate::Shape& launch_shape) {
  return core_runtime_->create_task(context_, task_id, launch_shape);
}

void Runtime::submit(std::unique_ptr<legate::Task> task) {
  core_runtime_->submit(std::move(task));
}

void Runtime::issue_execution_fence(bool block) {
  core_runtime_->issue_execution_fence(block);
}

/*static*/ Runtime* Runtime::get_runtime() {
  return runtime_;
}

/*static*/ void Runtime::initialize(legate::Runtime* core_runtime,
                                             legate::LibraryContext* context) {
  if (nullptr == runtime_)
    runtime_ = new Runtime(core_runtime, context);
}


/*static*/ void registration_callback(
    Legion::Machine machine, Legion::Runtime* legion_runtime,
    const std::set<Legion::Processor>& local_procs) {
  ResourceConfig config;
  config.max_tasks = 64;
  config.max_projections = 0;
  // We register one sharding functor for each new projection functor
  config.max_shardings = 0;
  config.max_reduction_ops = 0;

  auto runtime = legate::Runtime::get_runtime();

  auto context = runtime->create_library(library_name, config);

  LegateXla::get_registrar().register_all_tasks(*context);

  // Now we can register our mapper with the runtime
  context->register_mapper(std::make_unique<Mapper>(), 0);

  Runtime::initialize(runtime, context);
}

}  // namespace legate_xla

extern "C" {

void legate_jax_perform_registration() {
  Legion::Runtime::perform_registration_callback(
      legate_xla::registration_callback, true /*global*/);
}

}