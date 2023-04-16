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

#include "core/mapping/base_mapper.h"
#include "legate.h"
#include "xla_task.h"
#include "legate_xla.h"

namespace legate_xla {

static const char* library_name = "legate.jax";

// Simple runtime holding a pointer to the library context

struct Runtime {
 public:
  Runtime(legate::Runtime* core_runtime, legate::LibraryContext* context);

 public:
  legate::LibraryContext* get_context() const { return context_; }

 public:
  std::unique_ptr<legate::AutoTask> create_task(XlaOpCode task_id);
  std::unique_ptr<legate::ManualTask> create_task(XlaOpCode task_id,
                                          const legate::Shape& launch_shape);
  void submit(std::unique_ptr<legate::Task> task);
  void issue_execution_fence(bool block = false);
  std::vector<legate::LogicalStore>& get_tmp_stores();

 public:
  static Runtime* get_runtime();
  static void initialize(legate::Runtime* core_runtime, legate::LibraryContext* context);

 private:
  static Runtime* runtime_;

 private:
  legate::Runtime* core_runtime_;
  legate::LibraryContext* context_;
  std::vector<legate::LogicalStore> temporary_stores;
};

// Legate JAX mapper
class Mapper : public legate::mapping::LegateMapper {
 public:
  // LegateJAXMapper(Legion::Runtime* rt, Legion::Machine machine,
  //                 const legate::LibraryContext& context);
  Mapper();
  virtual ~Mapper(void) {}

 private:
  Mapper(const Mapper& rhs) = delete;
  Mapper& operator=(const Mapper& rhs) = delete;

  // Legate mapping functions
 public:
  void set_machine(
      const legate::mapping::MachineQueryInterface* machine) override;
  legate::mapping::TaskTarget task_target(
      const legate::mapping::Task& task,
      const std::vector<legate::mapping::TaskTarget>& options) override;
  std::vector<legate::mapping::StoreMapping> store_mappings(
      const legate::mapping::Task& task,
      const std::vector<legate::mapping::StoreTarget>& options) override;
  legate::Scalar tunable_value(legate::TunableID tunable_id) override;
};

// Registration callback for Legate JAX
/*static*/ void registration_callback(
    Legion::Machine machine, Legion::Runtime* legion_runtime,
    const std::set<Legion::Processor>& local_procs);

}  // namespace legate

extern "C" {
void legate_xla_perform_registration();
}

