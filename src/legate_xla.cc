#include "legate_xla.h"
#include "xla_task.h"

#ifndef LEGATE_XLA_PYTHON_PROTOTYPE

#include "legate_runtime.h"
#include <core/data/logical_store.h>

namespace legate_xla {

namespace {

int64_t GetRunId() {
    static std::atomic<int64_t> counter{0};
    return counter.fetch_add(1);
}

}

struct StoreHandleImpl : public StoreHandle {
  legate::LogicalStore store;
};

void CreateCompileTask(LegateCompiler* compiler){
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();

  auto task = runtime->create_task(XlaOpCode::XLA_EXECUTE_TASK);
  auto part = task->declare_partition();

  task->add_scalar_arg(legate::Scalar(compiler));
  task->add_scalar_arg(legate::Scalar(GetRunId()));
  // number of partitions
  task->add_scalar_arg(legate::Scalar(int64_t(1)));

  runtime->submit(std::move(task));
}

void CreateExecuteTask(
  LegateExecutable* executable,
  const std::vector<StoreHandle*>& inputs,
  const std::vector<StoreHandle*>& outputs){

  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();

  auto task = runtime->create_task(XlaOpCode::XLA_EXECUTE_TASK);
  auto part = task->declare_partition();

  task->add_scalar_arg(legate::Scalar(executable));
  task->add_scalar_arg(legate::Scalar(GetRunId()));

  for (auto& input : inputs){
    task->add_input(static_cast<StoreHandleImpl*>(input)->store, part);
  }

  for (auto& output: outputs){
    task->add_output(static_cast<StoreHandleImpl*>(output)->store, part);
  }

  runtime->submit(std::move(task));
}

}

#endif

