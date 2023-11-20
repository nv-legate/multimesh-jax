#include "legate_mapper.h"
#include "legate_to_xla.h"
#include "legate_xla_common.h"
#include "task_utils.h"
#include "xla_task.h"
#include "xla_to_legate.h"

#include <core/data/logical_store.h>
#include <core/data/scalar.h>
#include <core/mapping/mapping.h>
#include <core/task/task.h>
#include <numeric>
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
    if (shape.tile_shape[dim] == 0) {
      if (shape.dims[dim] != 0) {
        std::cerr << "Tile shape is zero, but dim is non-zero for " << shape
                  << std::endl;
        abort();
      }
    } else {
      size_t color_shape = shape.dims[dim] / shape.tile_shape[dim];
      size *= color_shape;
    }
  }
  return size;
}

struct get_read_only_ptr {
  template <legate::Type::Code TYPE_CODE, int32_t DIM>
  const void *operator()(legate::PhysicalStore &store) {
    using VAL = legate::type_of<TYPE_CODE>;
    auto shape = store.shape<DIM>();
    auto acc = store.read_accessor<VAL, DIM>();
    const void *buffer = static_cast<const void *>(acc.ptr(shape));
    return buffer;
  }
};

Shape ComputeStoreShape(const Shape &shape) {
  Shape store_shape{.type = shape.type, .replicated = 1};
  if (shape.replicated > 1) {
    store_shape.dims.push_back(shape.replicated);
    store_shape.dims.insert(store_shape.dims.end(), shape.dims.begin(),
                            shape.dims.end());
    store_shape.tile_shape.push_back(1);
  } else if (shape.dims.empty()) {
    store_shape.dims = {1};
    store_shape.tile_shape = {1};
    return store_shape;
  } else {
    store_shape.dims = shape.dims;
  }

  const auto &tile_shape =
      shape.tile_shape.empty() ? shape.dims : shape.tile_shape;
  store_shape.tile_shape.insert(store_shape.tile_shape.end(),
                                tile_shape.begin(), tile_shape.end());

  for (auto idx = 0; idx < store_shape.dims.size(); ++idx) {
    if (store_shape.dims[idx] == 0) {
      store_shape.dims[idx] = 1;
    }
    if (store_shape.tile_shape[idx] == 0) {
      store_shape.tile_shape[idx] = 1;
    }
  }
  return store_shape;
}

} // namespace

struct StoreHandleImpl {
  legate::LogicalStore store;
  Shape shape;
  std::optional<legate::LogicalStorePartition> partition;
  std::string name;
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

void CreateCompileTask(TaskArgHold<LegateCompiler> *compiler_hold) {
  auto *compiler = compiler_hold->get();
  LOCK;
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto machine = core_runtime->get_machine();
  auto [start, stop] = compiler->MachineSlice();
  legate::MachineTracker tracker(machine.slice(start, stop));
  log_xla.debug() << "CreateCompileTask " << compiler->Name()
                  << " scheduling on slice [" << start << "," << stop << ")";
  size_t launch_size = compiler->LaunchSize();
  if ((stop - start) < launch_size) {
    std::cerr << "Not enough devices to run launch size " << launch_size
              << " on task " << compiler->Name() << std::endl;
    abort();
  }
  auto task = runtime->create_task(XlaOpCode::XLA_COMPILE_TASK, {launch_size});

  task.add_scalar_arg(
      legate::Scalar(reinterpret_cast<uint64_t>(compiler_hold)));
  task.add_scalar_arg(legate::Scalar(GetRunId()));
  // number of partitions
  task.add_scalar_arg(legate::Scalar(int64_t(1)));

  runtime->submit(std::move(task));
}

void CreateExecuteTask(
    TaskArgHold<LegateCompiler> *compiler_hold, // *executable,
    const std::vector<StoreHandle> &inputs,
    const std::vector<StoreHandle> &outputs,
    std::vector<std::function<void()>> *on_done) {
  {
    auto *compiler = compiler_hold->get();

    size_t launch_size = compiler->LaunchSize();
    legate::Shape flattened({launch_size});

    LOCK;
    auto runtime = legate_xla::Runtime::get_runtime();
    auto core_runtime = legate::Runtime::get_runtime();
    auto machine = core_runtime->get_machine();
    auto [start, stop] = compiler->MachineSlice();
    legate::MachineTracker tracker(machine.slice(start, stop));
    log_xla.debug() << "CreateExecuteTask " << compiler->Name()
                    << " for launch shape " << flattened << " on slice ["
                    << start << "," << stop << ")";
    if (log_xla.want_debug()) {
      for (const auto &input : inputs) {
        log_xla.debug() << compiler->Name() << " has input " << input.impl->name
                        << " with shape " << input.impl->shape
                        << ", store=" << input.impl.get() << ", partitioned="
                        << input.impl->partition.has_value();
      }
      for (const auto &output : outputs) {
        log_xla.debug() << compiler->Name() << " has output "
                        << output.impl->name << " with shape "
                        << output.impl->shape << ", store=" << output.impl.get()
                        << ", partitioned="
                        << output.impl->partition.has_value();
      }
    }
    if ((stop - start) < launch_size) {
      std::cerr << "Not enough devices to run launch shape " << launch_size
                << " on task " << compiler->Name() << std::endl;
      abort();
    }
    auto task = runtime->create_task(XlaOpCode::XLA_EXECUTE_TASK, flattened);

    task.add_scalar_arg(
        legate::Scalar(reinterpret_cast<uint64_t>(compiler_hold)));
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
  }

  if (legate_xla::Runtime::synchronous_mode()) {
    for (const auto &output : outputs) {
      Synchronize(output);
    }
  }
}

void CopyDeviceToDevice(const StoreHandle &store, const void *src, size_t size,
                        size_t num_local_devices) {
  size_t launch_size = LaunchSize(store.impl->shape);
  log_xla.debug() << "CopyDeviceToDevice with launch size " << launch_size;

  LOCK;
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto task =
      runtime->create_task(XlaOpCode::XLA_COPY_DEVICE_TO_DEVICE, {launch_size});
  if (store.impl->partition.has_value()) {
    task.add_output(*store.impl->partition);
  } else {
    task.add_output(store.impl->store);
  }

  task.add_scalar_arg(reinterpret_cast<uint64_t>(src));
  task.add_scalar_arg(static_cast<uint64_t>(size));

  TaskWaiter waiter(num_local_devices);
  task.add_scalar_arg(reinterpret_cast<uint64_t>(&waiter));

  runtime->submit(std::move(task));
  waiter.Wait();
}

void Destroy(StoreHandle &store) {
  LOCK;
  log_xla.debug() << "Destroy store";
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

void BufferFromHostBuffer(BufferAction *action, StoreHandle store,
                          int num_local_devices, bool blocking) {
  size_t launch_size = LaunchSize(store.impl->shape);
  log_xla.debug() << "legate_xla::BufferFromHostBuffer with launch size "
                  << launch_size << " num_local_devices=" << num_local_devices;

  LOCK;
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto task = runtime->create_task(XlaOpCode::XLA_BUFFER_FROM_HOST_BUFFER_TASK,
                                   {launch_size});

  TaskWaiter waiter(num_local_devices);
  task.add_scalar_arg(reinterpret_cast<uint64_t>(action));
  task.add_scalar_arg(blocking);
  if (blocking) {
    task.add_scalar_arg(reinterpret_cast<uint64_t>(&waiter));
  }

  if (store.impl->partition.has_value()) {
    task.add_output(*store.impl->partition);
  } else {
    task.add_output(store.impl->store);
  }
  runtime->submit(std::move(task));

  if (blocking) {
    waiter.Wait();
  }
}

void SliceLocalShards(const StoreHandle &handle,
                      std::vector<void *> &local_shards,
                      const std::vector<size_t> &devices) {
  LOCK;
  size_t start = devices.front();
  size_t stop = devices.back() + 1;
  size_t check = start;
  for (auto device_id : devices) {
    if (check != device_id) {
      std::cerr << "Legate received non-contiguous device slice" << std::endl;
      abort();
    }
    check++;
  }

  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto task = runtime->create_task(XlaOpCode::XLA_SHARD_GETTER_TASK,
                                   {LaunchSize(handle.impl->shape)});

  TaskWaiter waiter{int64_t(local_shards.size())};
  task.add_scalar_arg(reinterpret_cast<uint64_t>(local_shards.data()));
  task.add_scalar_arg(int64_t(local_shards.size()));
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
    Shape store_shape = ComputeStoreShape(new_shape);
    auto new_impl = std::make_shared<StoreHandleImpl>(
        StoreHandleImpl{.store = handle.impl->store, .shape = new_shape});
    new_impl->partition =
        new_impl->store.partition_by_tiling(store_shape.tile_shape);
    return StoreHandle{.impl = std::move(new_impl)};
  }
  // just return back the original handle, no resharding
  return handle;
}

bool IsGpu() {
  auto core_runtime = legate::Runtime::get_runtime();
  auto target = core_runtime->get_machine().preferred_target();
  return target == legate::mapping::TaskTarget::GPU;
}

std::set<int> GetLocalDevices(int my_node) {
  LOCK;
  auto core_runtime = legate::Runtime::get_runtime();
  const auto &range = core_runtime->get_machine().processor_range();
  int num_gpus = range.high - range.low;
  int num_nodes = range.get_node_range().high - range.get_node_range().low;
  int gpus_per_node = num_gpus / num_nodes;
  int my_gpu_start = gpus_per_node * my_node;
  std::set<int> gpus;
  // these are numbered locally starting from zero
  for (int i = 0; i < gpus_per_node; ++i) {
    gpus.insert(i);
  }
  return gpus;
}

StoreHandle CreateStore(const legate_xla::Shape &shape,
                        std::optional<std::string> name) {
  legate::Type::Code code = SupportedTypeToLegateType(shape.type);

  auto store_shape = ComputeStoreShape(shape);
  bool is_scalar = false;

  log_xla.debug() << "CreateStore " << (name.has_value() ? *name : "")
                  << ": logical=" << shape << ", actual=" << store_shape;
  LOCK;
  auto core_runtime = legate::Runtime::get_runtime();
  StoreHandle result = {
      .impl = std::make_shared<StoreHandleImpl>(
          StoreHandleImpl{.store = core_runtime->create_store(
                              legate::Shape(std::move(store_shape.dims)),
                              legate::primitive_type(code), is_scalar),
                          .shape = shape})};

  if (result.impl->shape.tile_shape.empty()) {
    result.impl->shape.tile_shape = shape.dims;
  }

  if (name.has_value()) {
    result.impl->name = *name;
  }

  if (!store_shape.tile_shape.empty()) {
    result.impl->partition =
        result.impl->store.partition_by_tiling(store_shape.tile_shape);
  }

  if (legate_xla::Runtime::synchronous_mode()) {
    Synchronize(result);
  }

  return result;
}

void SetScalar(legate_xla::StoreHandle handle, size_t launch_size,
               int32_t scalar) {
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto task =
      runtime->create_task(XlaOpCode::XLA_SET_SCALAR_TASK, {launch_size});
  legate::Scalar wtf(scalar);
  task.add_scalar_arg(scalar);
  if (handle.impl->partition.has_value()) {
    task.add_output(*handle.impl->partition);
  } else {
    task.add_output(handle.impl->store);
  }
  runtime->submit(std::move(task));
}

enum LegateState { UNINITIALIZED, STARTING, STARTED, STOPPING, STOPPED };
static std::atomic<int> legate_state{UNINITIALIZED};

void StopLegate() {
  int started = STARTED;
  if (legate_state.compare_exchange_strong(started, int(STOPPING))) {
    log_xla.info() << "Stopping Legate";
    auto rc = legate::finish();
    legate_state = STOPPED;
  }
}

void StartLegate() {
  int not_started = UNINITIALIZED;
  if (legate_state.compare_exchange_strong(not_started, int(STARTING))) {
    log_xla.info() << "Starting Legate";
    auto rc = legate::start(0, nullptr);
    legate_xla_perform_registration();
    legate_state = STARTED;
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
