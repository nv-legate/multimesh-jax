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

#pragma once

#include "legate.h"
#include "legate_xla_c.h"
#include "legate_xla_common.h"
#include "xla_task.h"
#include <mutex>

namespace legate_xla {

// Simple runtime holding a pointer to the library context

struct Runtime {
public:
  Runtime(legate::Runtime *core_runtime, legate::Library context);

public:
  legate::AutoTask create_task(XlaOpCode task_id);
  legate::ManualTask create_task(XlaOpCode task_id,
                                 const legate::Shape &launch_shape);
  void submit(legate::AutoTask task);
  void submit(legate::ManualTask task);
  void issue_execution_fence(bool block = false);
  std::vector<legate::LogicalStore> &get_tmp_stores();

public:
  static Runtime *get_runtime();
  static void initialize(legate::Runtime *core_runtime,
                         legate::Library library);
  static std::mutex &get_mutex();

private:
  static Runtime *runtime_;
  static std::mutex mutex_;

private:
  legate::Runtime *core_runtime_;
  legate::Library context_;
  std::vector<legate::LogicalStore> temporary_stores;
};

#define LOCK                                                                   \
  const std::lock_guard<std::mutex> lock(legate_xla::Runtime::get_mutex())

} // namespace legate_xla
