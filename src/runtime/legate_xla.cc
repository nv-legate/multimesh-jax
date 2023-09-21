#include "legate_mapper.h"
#include "legate_to_xla.h"
#include "legate_xla_common.h"
#include "task_utils.h"
#include "xla_task.h"
#include "xla_to_legate.h"

#include <core/data/logical_store.h>
#include <core/task/task.h>
#include <tuple>
#include <unistd.h>
#include <utility>

#include "legate_runtime.h"

namespace legate_xla {
namespace {

template <class T> struct RefCountScalarArg {
  T arg;
  std::atomic<int> refcount{0};
};

int64_t GetRunId() {
  static std::atomic<int64_t> counter{0};
  return counter.fetch_add(1);
}

size_t LaunchSize(const Shape &shape) {
  size_t size = shape.replicated;
  for (size_t dim = 0; dim < shape.dims.size(); ++dim) {
    size_t color_shape = shape.dims[dim] / shape.tile_shape[dim];
    size *= color_shape;
  }
  return size;
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

struct ReplicatedStoreShape {
  size_t replication = 1;
  std::vector<size_t> dims;
  std::vector<size_t> tile_shape;
};

ReplicatedStoreShape ComputeStoreShape(const Shape &shape) {
  std::vector<size_t> dims;
  std::vector<size_t> tile_shape;
  ReplicatedStoreShape store_shape{.replication = shape.replicated};
  if (shape.replicated > 1) {
    store_shape.dims.push_back(shape.replicated);
    store_shape.dims.insert(dims.end(), shape.dims.begin(), shape.dims.end());
    store_shape.tile_shape.push_back(1);
    store_shape.tile_shape.insert(tile_shape.end(), shape.tile_shape.begin(),
                                  shape.tile_shape.end());
  } else {
    store_shape.dims = shape.dims;
    store_shape.tile_shape = shape.tile_shape;
  }
  return store_shape;
}

} // namespace

struct StoreHandleImpl {
  legate::LogicalStore store;
  Shape shape;
  std::optional<legate::LogicalStorePartition> partition;
};

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

size_t SupportedTypeSizeOf(SupportedType type) {
  size_t bytesize = 0;
  switch (type) {
  case SupportedType::PRED:
  case SupportedType::S8:
  case SupportedType::U8:
    bytesize = 1;
    break;
  case SupportedType::F16:
  case SupportedType::BF16:
  case SupportedType::S16:
  case SupportedType::U16:
    bytesize = 2;
    break;
  case SupportedType::F32:
  case SupportedType::S32:
  case SupportedType::U32:
    bytesize = 4;
    break;
  case SupportedType::F64:
  case SupportedType::S64:
  case SupportedType::U64:
  case SupportedType::C64:
    bytesize = 8;
    break;
  case SupportedType::C128:
    bytesize = 16;
    break;
  default:
    std::cerr << "Unsupported SupportedType type "
              << static_cast<std::underlying_type<SupportedType>::type>(type)
              << std::endl;
    break;
  }
  return bytesize;
}

size_t ShapeNumElements(Shape shape) {
  if (shape.dims.empty()) {
    // this is a scalar
    return 1ul;
  } else {
    return std::accumulate(begin(shape.dims), end(shape.dims), 1,
                           std::multiplies<size_t>());
  }
}

void CreateCompileTask(LegateCompiler *compiler) {
  LOCK;
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();

  legate::Shape launch_shape = compiler->LaunchShape();
  auto task = runtime->create_task(XlaOpCode::XLA_COMPILE_TASK, launch_shape);

  task.add_scalar_arg(legate::Scalar(reinterpret_cast<uint64_t>(compiler)));
  task.add_scalar_arg(legate::Scalar(GetRunId()));
  // number of partitions
  task.add_scalar_arg(legate::Scalar(int64_t(1)));

  runtime->submit(std::move(task));
  log_xla.debug() << "CreateCompileTask scheduled";
}

void CreateExecuteTask(LegateExecutable *executable,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       std::vector<std::function<void()>> *on_done) {
  {
    LOCK;
    auto runtime = legate_xla::Runtime::get_runtime();
    auto core_runtime = legate::Runtime::get_runtime();

    legate::Shape launch_shape(executable->LaunchShape());
    legate::Shape flattened({launch_shape.volume()});

    auto task = runtime->create_task(XlaOpCode::XLA_EXECUTE_TASK, flattened);

    task.add_scalar_arg(legate::Scalar(reinterpret_cast<uint64_t>(executable)));
    task.add_scalar_arg(legate::Scalar(GetRunId()));

    task.add_scalar_arg(
        legate::Scalar(reinterpret_cast<uint64_t>(std::move(on_done))));

    for (const auto &input : inputs) {
      if (input.impl->partition.has_value()) {
        task.add_input(*input.impl->partition);
      } else {
        task.add_input(input.impl->store);
      }
    }

    for (const auto &output : outputs) {
      if (output.impl->partition.has_value()) {
        task.add_output(*output.impl->partition);
      } else {
        task.add_output(output.impl->store);
      }
      task.add_scalar_arg(legate::Scalar(false));
    }

    runtime->submit(std::move(task));
    log_xla.debug() << "CreateExecuteTask scheduled";
  }

  if (legate_xla::Runtime::synchronous_mode()) {
    for (const auto &output : outputs) {
      Synchronize(output);
    }
  }
}

void CreateStoreFromHostBufferTask(const void *data, uint64_t num_bytes,
                                   StoreHandle &output,
                                   std::function<void()> on_done) {
  {
    LOCK;
    auto runtime = legate_xla::Runtime::get_runtime();
    auto core_runtime = legate::Runtime::get_runtime();

    auto task = runtime->create_task(XlaOpCode::XLA_INIT_FROM_HOST_TASK);
    auto part = task.declare_partition();
    task.add_output(output.impl->store, part);
    task.add_scalar_arg(legate::Scalar(static_cast<uint64_t>(num_bytes)));
    task.add_scalar_arg(legate::Scalar(reinterpret_cast<uint64_t>(data)));

    if (on_done) {
      task.add_scalar_arg(legate::Scalar(true));
      auto on_done_copy =
          std::make_unique<std::function<void()>>(std::move(on_done));
      auto arg = reinterpret_cast<uint64_t>(on_done_copy.release());
      task.add_scalar_arg(legate::Scalar(arg));
    } else {
      task.add_scalar_arg(legate::Scalar(false));
    }

    runtime->submit(std::move(task));
  }
  log_xla.debug() << "CreateStoreFromHostBufferTask store " << output.impl
                  << " scheduled";

  if (legate_xla::Runtime::synchronous_mode()) {
    Synchronize(output);
  }
}

void Destroy(StoreHandle &store) {
  LOCK;
  log_xla.debug() << "Destroy Store";
  // no need to synchronize -- just removing the reference
  store.impl = nullptr;
}

void Synchronize(const StoreHandle &store) {
  LOCK;
  log_xla.debug() << "Synchronize store " << store.impl << " start";
  auto runtime = legate_xla::Runtime::get_runtime();
  auto logical_store = store.impl->store;
  auto out_mapped = logical_store.get_physical_store();
  auto buffer_alloc = legate::double_dispatch(
      out_mapped.dim(), out_mapped.code(), get_read_only_ptr{}, out_mapped);
  log_xla.debug() << "Synchronize store " << store.impl << " done";
}

void SliceLocalShards(const StoreHandle &handle,
                      std::vector<void *> &local_shards) {
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto task = runtime->create_task(XlaOpCode::XLA_SHARD_GETTER_TASK,
                                   {LaunchSize(handle.impl->shape)});

  TaskWaiter waiter{int64_t(local_shards.size())};
  task.add_scalar_arg(reinterpret_cast<uint64_t>(local_shards.data()));
  task.add_scalar_arg(reinterpret_cast<uint64_t>(&waiter));

  // Treat this as an output for future synchronization purposes
  if (handle.impl->partition.has_value()) {
    task.add_input(*handle.impl->partition);
  } else {
    task.add_input(handle.impl->store);
  }
  runtime->submit(std::move(task));

  waiter.Wait();
}

StoreHandle Reshard(const StoreHandle &handle,
                    const std::vector<size_t> &tile_shape) {
  if (tile_shape != handle.impl->shape.tile_shape) {
    Shape new_shape = handle.impl->shape;
    new_shape.tile_shape = tile_shape;
    ReplicatedStoreShape store_shape = ComputeStoreShape(new_shape);
    auto new_impl = std::make_shared<StoreHandleImpl>(
        StoreHandleImpl{.store = handle.impl->store, .shape = new_shape});
    new_impl->partition =
        new_impl->store.partition_by_tiling(store_shape.tile_shape);
    return StoreHandle{.impl = std::move(new_impl)};
  }
  // just return back the original handle, no resharding
  return handle;
}

StoreHandle CreateStore(const legate_xla::Shape &shape) {
  legate::Type::Code code = SupportedTypeToLegateType(shape.type);

  auto store_shape = ComputeStoreShape(shape);

  bool is_scalar = false;

  auto core_runtime = legate::Runtime::get_runtime();

  LOCK;
  StoreHandle result = {
      .impl = std::make_shared<StoreHandleImpl>(StoreHandleImpl{
          .store = core_runtime->create_store(
              store_shape.dims, legate::primitive_type(code), is_scalar),
          .shape = shape})};

  if (!store_shape.tile_shape.empty()) {
    result.impl->partition =
        result.impl->store.partition_by_tiling(store_shape.tile_shape);
  }

  if (legate_xla::Runtime::synchronous_mode()) {
    Synchronize(result);
  }

  log_xla.debug() << "CreateStore " << result.impl << " done with " << shape;
  return result;
}

enum LegateState { UNINITIALIZED, STARTING, STARTED, STOPPING, STOPPED };
static std::atomic<int> legate_state{UNINITIALIZED};

void StartLegate() {

  static int not_started = UNINITIALIZED;
  if (legate_state.compare_exchange_strong(not_started, int(STARTING))) {
    log_xla.info() << "Starting Legate";
    legate::start(0, nullptr);
    legate_xla_perform_registration();
    legate_state = STARTED;
  }
}

void StopLegate() {

  static int started = STARTED;
  if (legate_state.compare_exchange_strong(started, int(STOPPING))) {
    log_xla.info() << "Stopping Legate";
    legate::finish();
    legate_state = STOPPED;
  }
}

std::ostream &operator<<(std::ostream &os, const Shape &shape) {
  std::ostringstream ss;
  ss << "Shape(type="
     << static_cast<std::underlying_type<SupportedType>::type>(shape.type)
     << ",dim=[";
  for (auto i : shape.dims)
    ss << i << ",";
  ss << "]),tile=[";
  for (auto i : shape.tile_shape)
    ss << i << ",";
  ss << "], replication=" << shape.replicated;
  os << ss.str();
  return os;
}

void initialize_runtime_and_context(legate::Runtime *runtime,
                                    legate::Library context) {

  legate_xla::Runtime::initialize(runtime, context);
}

} // namespace legate_xla

extern "C" {

struct PJRT_Api;
const PJRT_Api *GetPjrtApi() { return GetLegatePjrtApi(); }
}
