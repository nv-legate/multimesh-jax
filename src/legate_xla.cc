#include "legate_mapper.h"
#include "legate_xla_common.h"
#include "xla_task.h"
#include "xla_to_legate.h"

#ifndef LEGATE_XLA_PYTHON_PROTOTYPE
#include <core/data/logical_store.h>

#include "legate_runtime.h"

namespace legate_xla {
namespace {

int64_t GetRunId() {
  static std::atomic<int64_t> counter{0};
  return counter.fetch_add(1);
}

legate::Type::Code SupportedTypeToLegateType(SupportedType type) {
  legate::Type::Code code;
  switch (type) {
  case SupportedType::PRED:
    code = legate::Type::Code::BOOL;
    break;
  case SupportedType::S8:
    code = legate::Type::Code::INT8;
    break;
  case SupportedType::U8:
    code = legate::Type::Code::UINT8;
    break;
  case SupportedType::F16:
    /* fall-through */
  case SupportedType::BF16:
    code = legate::Type::Code::FLOAT16;
    break;
  case SupportedType::S16:
    code = legate::Type::Code::INT16;
    break;
  case SupportedType::U16:
    code = legate::Type::Code::UINT16;
    break;
  case SupportedType::F32:
    code = legate::Type::Code::FLOAT32;
    break;
  case SupportedType::S32:
    code = legate::Type::Code::INT32;
    break;
  case SupportedType::U32:
    code = legate::Type::Code::UINT32;
    break;
  case SupportedType::F64:
    code = legate::Type::Code::FLOAT64;
    break;
  case SupportedType::S64:
    code = legate::Type::Code::INT64;
    break;
  case SupportedType::U64:
    code = legate::Type::Code::UINT64;
    break;
  case SupportedType::C64:
    code = legate::Type::Code::COMPLEX64;
    break;
  case SupportedType::C128:
    code = legate::Type::Code::COMPLEX128;
    break;
  default:
    std::cerr << "Unsupported SupportedType type "
              << static_cast<std::underlying_type<SupportedType>::type>(type)
              << std::endl;
    break;
  }
  return code;
}

struct get_read_only_ptr {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  const void *operator()(legate::Store &store) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.read_accessor<VAL, DIM>();
    const void *buffer = static_cast<const void *>(acc.ptr(shape));
    return buffer;
  }
};

} // namespace

struct StoreHandleImpl {
  legate::LogicalStore store;
};

void CreateCompileTask(LegateCompiler *compiler) {
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();

  auto task = runtime->create_task(XlaOpCode::XLA_COMPILE_TASK);
  auto part = task->declare_partition();

  task->add_scalar_arg(legate::Scalar(reinterpret_cast<uint64_t>(compiler)));
  task->add_scalar_arg(legate::Scalar(GetRunId()));
  // number of partitions
  task->add_scalar_arg(legate::Scalar(int64_t(1)));

  auto sync_store_handle = compiler->SyncStoreHandle();

  task->add_output(sync_store_handle.impl->store, part);

  runtime->submit(std::move(task));

  if (legate_xla::Runtime::synchronous_mode()) {
    Synchronize(sync_store_handle);
  }
}

void CreateExecuteTask(LegateExecutable *executable,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       std::vector<std::function<void()>> *on_done) {
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();

  auto task = runtime->create_task(XlaOpCode::XLA_EXECUTE_TASK);

  task->add_scalar_arg(legate::Scalar(reinterpret_cast<uint64_t>(executable)));
  task->add_scalar_arg(legate::Scalar(GetRunId()));

  task->add_scalar_arg(
      legate::Scalar(reinterpret_cast<uint64_t>(std::move(on_done))));

  for (auto input : inputs) {
    task->add_input(input.impl->store,
                    task->find_or_declare_partition(input.impl->store));
  }

  for (auto output : outputs) {
    task->add_output(output.impl->store,
                     task->find_or_declare_partition(output.impl->store));
    task->add_scalar_arg(legate::Scalar(false));
  }

  runtime->submit(std::move(task));

  if (legate_xla::Runtime::synchronous_mode()) {
    for (auto output : outputs) {
      Synchronize(output);
    }
  }
}

void CreateStoreFromHostBufferTask(const void *data, uint64_t num_bytes,
                                   StoreHandle output,
                                   std::function<void()> on_done) {
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();

  auto task = runtime->create_task(XlaOpCode::XLA_INIT_FROM_HOST_TASK);
  auto part = task->declare_partition();
  task->add_output(output.impl->store, part);
  task->add_scalar_arg(legate::Scalar(static_cast<uint64_t>(num_bytes)));
  task->add_scalar_arg(legate::Scalar(reinterpret_cast<uint64_t>(data)));

  if (on_done) {
    task->add_scalar_arg(legate::Scalar(true));
    auto on_done_copy = std::make_unique<std::function<void()>>(on_done);
    task->add_scalar_arg(
        legate::Scalar(reinterpret_cast<uint64_t>(on_done_copy.release())));
  } else {
    task->add_scalar_arg(legate::Scalar(false));
  }

  runtime->submit(std::move(task));
  log_xla.debug() << "CreateStoreFromHostBufferTask store " << output.impl
                  << " scheduled";

  if (legate_xla::Runtime::synchronous_mode()) {
    Synchronize(output);
  }
}

void Destroy(StoreHandle store) {
  log_xla.debug() << "Destroy Store " << store.impl;
  delete store.impl;
}

void Synchronize(StoreHandle store) {
  log_xla.debug() << "Synchronize store " << store.impl << " start";
  auto runtime = legate_xla::Runtime::get_runtime();
  auto logical_store = store.impl->store;
  auto out_mapped = logical_store.get_physical_store();
  auto buffer_alloc = legate::double_dispatch(
      out_mapped->dim(), out_mapped->code(), get_read_only_ptr{}, *out_mapped);
  log_xla.debug() << "Synchronize store " << store.impl << " done";
}

void CopyStoreToHostSync(StoreHandle input,
                         std::function<void(const void *)> copy_func) {
  log_xla.debug() << "CopyStoreToHostSync " << input.impl << " start";
  auto runtime = legate_xla::Runtime::get_runtime();
  auto logical_store = input.impl->store;
  auto out_mapped = logical_store.get_physical_store();
  auto buffer_alloc = legate::double_dispatch(
      out_mapped->dim(), out_mapped->code(), get_read_only_ptr{}, *out_mapped);
  copy_func(buffer_alloc);
  log_xla.debug() << "CopyStoreToHostSync " << input.impl << " done";
}

StoreHandle CreateStore(const legate_xla::Shape &shape) {
  legate::Type::Code code = SupportedTypeToLegateType(shape.type);

  // legate modification of dimensions:
  // 1. handle scalars (dims.size() == 0) as arrays of size 1
  // 2. squash all dimensions >= 4
  // 3. if any dimension has size 0 we create a scalar store
  // FIXME: evaluate MAX_DIM legion
  std::vector<size_t> dims(shape.dims);
  bool is_scalar = false;
  if (dims.empty()) {
    is_scalar = true;
    dims.push_back(1ul);
  } else if (dims.size() > 4) {
    log_xla.warning() << "Number of dimensions(" << dims.size()
                      << ") > 4! Collapse all dims > 3 to 4!";
    uint64_t collapsed_4th = 1ul;
    while (dims.size() >= 4) {
      collapsed_4th *= dims.back();
      dims.pop_back();
    }
    dims.push_back(collapsed_4th);
  }

  auto core_runtime = legate::Runtime::get_runtime();

  auto total_elements =
      std::accumulate(begin(dims), end(dims), 1, std::multiplies<size_t>());
  if (total_elements > 0) {
    StoreHandle result = {
        .impl = new StoreHandleImpl{
            .store = core_runtime->create_store(
                dims, legate::primitive_type(code), is_scalar)}};
    log_xla.debug() << "CreateStore " << result.impl << " done with " << shape;
    return result;
  } else {
    log_xla.debug()
        << "Create single element store to prevent empty allocation";
    StoreHandle result = {
        .impl = new StoreHandleImpl{.store = core_runtime->create_store(
                                        {1}, legate::primitive_type(code))}};

    // FIXME: initialize it with 0 (on host)
    // otherwise runtime might complain when read-accessing later
    {
      auto runtime = legate_xla::Runtime::get_runtime();
      auto task = runtime->create_task(XlaOpCode::XLA_INIT_ZERO_TASK);
      auto part = task->declare_partition();
      task->add_output(result.impl->store, part);
      runtime->submit(std::move(task));

      if (legate_xla::Runtime::synchronous_mode()) {
        Synchronize(result);
      }
    }

    log_xla.debug() << "CreateStore " << result.impl << " done with " << shape;
    return result;
  }
}

void InitLegate() {
  legate::start(0, nullptr);
  legate_xla_perform_registration();
}

std::ostream &operator<<(std::ostream &os, const Shape &shape) {
  std::ostringstream ss;
  ss << "Shape(type="
     << static_cast<std::underlying_type<SupportedType>::type>(shape.type)
     << ",dim=[";
  for (auto i : shape.dims)
    ss << i << ",";
  ss << "])";
  os << ss.str();
  return os;
}

} // namespace legate_xla
#else

namespace legate_xla {

void Synchronize(StoreHandle store) {
  /** never called, but needed to provide symbol */
}

StoreHandle CreateStore(const legate_xla::Shape &shape) {
  /** never called, but needed to provide symbol */
  return StoreHandle{};
}

} // namespace legate_xla
#endif

namespace legate_xla {
namespace {
static constexpr char library_name[] = "legate.xla";
}

/*static*/ void registration_callback() {
  legate::ResourceConfig config;
  config.max_tasks = 64;
  config.max_projections = 0;
  // We register one sharding functor for each new projection functor
  config.max_shardings = 0;
  config.max_reduction_ops = 0;

  auto runtime = legate::Runtime::get_runtime();

  auto context =
      runtime->create_library(library_name, config, std::make_unique<Mapper>());
#ifndef LEGATE_XLA_PYTHON_PROTOTYPE
  legate_xla::Runtime::initialize(runtime, context);
#endif

  Registry::get_registrar().register_all_tasks(context);
}

} // namespace legate_xla

extern "C" {

void legate_xla_perform_registration() {
  legate::Core::perform_registration<&legate_xla::registration_callback>();
}
}
