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

legate::LegateTypeCode SupportedTypeToLegateType(SupportedType type) {
  legate::LegateTypeCode code;
  switch (type) {
    case SupportedType::PRED:
      code = legate::LegateTypeCode::BOOL_LT;
      break;
    case SupportedType::S8:
      code = legate::LegateTypeCode::INT8_LT;
      break;
    case SupportedType::U8:
      code = legate::LegateTypeCode::UINT8_LT;
      break;
    case SupportedType::F16:
      /* fall-through */
    case SupportedType::BF16:
      code = legate::LegateTypeCode::HALF_LT;
      break;
    case SupportedType::S16:
      code = legate::LegateTypeCode::INT16_LT;
      break;
    case SupportedType::U16:
      code = legate::LegateTypeCode::UINT16_LT;
      break;
    case SupportedType::F32:
      code = legate::LegateTypeCode::FLOAT_LT;
      break;
    case SupportedType::S32:
      code = legate::LegateTypeCode::INT32_LT;
      break;
    case SupportedType::U32:
      code = legate::LegateTypeCode::UINT32_LT;
      break;
    case SupportedType::F64:
      code = legate::LegateTypeCode::DOUBLE_LT;
      break;
    case SupportedType::S64:
      code = legate::LegateTypeCode::INT64_LT;
      break;
    case SupportedType::U64:
      code = legate::LegateTypeCode::UINT64_LT;
      break;
    case SupportedType::C64:
      code = legate::LegateTypeCode::COMPLEX64_LT;
      break;
    case SupportedType::C128:
      code = legate::LegateTypeCode::COMPLEX128_LT;
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
  template <legate::LegateTypeCode TYPE_CODE, int32_t DIM>
  const void* operator()(legate::Store& store) {
    using VAL = legate::legate_type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.read_accessor<VAL, DIM>();
    const void* buffer = static_cast<const void*>(acc.ptr(shape));
    return buffer;
  }
};

}  // namespace

struct StoreHandleImpl {
  legate::LogicalStore store;
};

void CreateCompileTask(LegateCompiler* compiler) {
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
}

void CreateExecuteTask(LegateExecutable* executable,
                       const std::vector<StoreHandle>& inputs,
                       const std::vector<StoreHandle>& outputs,
                       std::vector<std::function<void()>>* on_done) {
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();

  auto task = runtime->create_task(XlaOpCode::XLA_EXECUTE_TASK);
  auto part = task->declare_partition();

  task->add_scalar_arg(legate::Scalar(reinterpret_cast<uint64_t>(executable)));
  task->add_scalar_arg(legate::Scalar(GetRunId()));

  task->add_scalar_arg(
      legate::Scalar(reinterpret_cast<uint64_t>(std::move(on_done))));

  for (auto input : inputs) {
    task->add_input(input.impl->store, part);
  }

  for (auto output : outputs) {
    task->add_output(output.impl->store, part);
    task->add_scalar_arg(legate::Scalar(false));
  }

  runtime->submit(std::move(task));
}

void CreateStoreFromHostBufferTask(const void* data, uint64_t num_bytes,
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
}

void Destroy(StoreHandle store) { delete store.impl; }

void Synchronize(StoreHandle store) {
  log_xla.debug() << "Synchronize called.";
  auto runtime = legate_xla::Runtime::get_runtime();
  auto logical_store = store.impl->store;
  auto out_mapped = logical_store.get_physical_store(runtime->get_context());
  auto buffer_alloc = legate::double_dispatch(
      out_mapped->dim(), out_mapped->code(), get_read_only_ptr{}, *out_mapped);
}

void CopyStoreToHostSync(StoreHandle input,
                         std::function<void(const void*)> copy_func) {
  log_xla.debug() << "CopyStoreToHostSync called.";
  auto runtime = legate_xla::Runtime::get_runtime();
  auto logical_store = input.impl->store;
  auto out_mapped = logical_store.get_physical_store(runtime->get_context());
  auto buffer_alloc = legate::double_dispatch(
      out_mapped->dim(), out_mapped->code(), get_read_only_ptr{}, *out_mapped);
  copy_func(buffer_alloc);
}

StoreHandle CreateStore(const legate_xla::Shape& shape) {
  legate::LegateTypeCode code = SupportedTypeToLegateType(shape.type);

  log_xla.debug() << "CreateStore called with " << shape;
  // legate modification of dimensions:
  // 1. handle scalars as arrays of size 1
  // 2. squash all dimensions >= 4
  // FIXME: evaluate MAX_DIM legion
  std::vector<size_t> dims(shape.dims);
  if (dims.empty()) {
    dims.push_back(1ul);
  } else if (dims.size() > 4) {
    log_xla.debug() << "Number of dimensions(" << dims.size()
                    << ") > 4! Collapse all dims > 3 to 4!";
    uint64_t collapsed_4th = 1ul;
    while (dims.size() >= 4) {
      collapsed_4th *= dims.back();
      dims.pop_back();
    }
    dims.push_back(collapsed_4th);
  }

  auto core_runtime = legate::Runtime::get_runtime();

  return {.impl = new StoreHandleImpl{
              .store = core_runtime->create_store(dims, code)}};
}

void InitLegate() {
  legate_parse_config();
  legate_core_perform_registration();
  legate::Runtime::get_runtime()->post_startup_initialization(
      Legion::Runtime::get_context());
  legate_xla_perform_registration();
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  std::ostringstream ss;
  ss << "Shape(type="
     << static_cast<std::underlying_type<SupportedType>::type>(shape.type)
     << ",dim=[";
  for (auto i : shape.dims) ss << i << ",";
  ss << "])";
  os << ss.str();
  return os;
}

}  // namespace legate_xla
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

#ifndef LEGATE_XLA_PYTHON_PROTOTYPE
  auto context =
      runtime->create_library(library_name, config, std::make_unique<Mapper>());
#else
  auto context = runtime->create_library(library_name, config);
  context->register_mapper(std::make_unique<Mapper>());
#endif

  legate_xla::Runtime::initialize(runtime, context);

  Registry::get_registrar().register_all_tasks(*context);
}

}  // namespace legate_xla

extern "C" {

void legate_xla_perform_registration() {
  legate::Core::perform_registration<&legate_xla::registration_callback>();
}
}
