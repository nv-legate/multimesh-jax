#include "legate_mapper.h"
#include "legate_to_xla.h"
#include "legate_xla_c.h"
#include "legate_xla_common.h"
#include "task_utils.h"
#include "xla_task.h"
#include "xla_to_legate.h"

#include <core/data/external_allocation.h>
#include <core/data/logical_store.h>
#include <core/data/scalar.h>
#include <core/mapping/mapping.h>
#include <core/task/task.h>
#include <core/type/type_info.h>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
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

size_t LaunchSize(const Shape &shape, size_t default_size) {
  if (shape.replicated) {
    return default_size;
  }

  size_t size = shape.replicated;
  for (size_t dim = 0; dim < shape.dims.size(); ++dim) {
    if (shape.tile_shape.has_value()) {
      size_t color_shape = shape.dims[dim] / (*shape.tile_shape)[dim];
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

Shape ComputeStoreShape(const Shape &shape, size_t global_size) {
  int64_t dim_product = 1;
  for (auto dim : shape.dims) {
    dim_product *= dim;
  }
  bool scalar = dim_product <= 1;

  Shape store_shape{.type = shape.type, .replicated = 1};

  if (scalar) {
    // ignore replication on scalars
    store_shape.dims = {1};
    return store_shape;
  }

  if (shape.replicated > 1) {
    store_shape.dims.push_back(shape.replicated);
    store_shape.dims.insert(store_shape.dims.end(), shape.dims.begin(),
                            shape.dims.end());
    store_shape.tile_shape = {1};
    if (shape.tile_shape.has_value()) {
      store_shape.tile_shape->insert(store_shape.tile_shape->end(),
                                     shape.tile_shape->begin(),
                                     shape.tile_shape->end());
    } else {
      store_shape.tile_shape->insert(store_shape.tile_shape->end(),
                                     shape.dims.begin(), shape.dims.end());
    }
  } else if (dim_product <= 1) {
    store_shape.dims = {1};
  } else {
    store_shape.dims = shape.dims;
    store_shape.tile_shape = shape.tile_shape;
  }

  return store_shape;
}

template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[";
  for (auto v : vec) {
    os << v << ",";
  }
  os << "]";
  return os;
}

} // namespace

struct StoreHandleImpl {

  StoreHandleImpl(legate::LogicalStore store, Shape shape, std::string name)
      : store_(store), shape_(shape), name_(name) {}

  ~StoreHandleImpl() = default;

  const Shape &shape() const { return shape_; }

  const legate::LogicalStore store() const { return store_; }

  legate::LogicalStore &store() { return store_; }

  bool HasPartition() const { return partition_.has_value(); }

  const legate::LogicalStorePartition &partition() const { return *partition_; }

  void SetPartition(legate::LogicalStorePartition partition) {
    partition_ = std::move(partition);
  }

  std::string name() const { return name_; }

private:
  legate::LogicalStore store_;
  Shape shape_;
  std::optional<legate::LogicalStorePartition> partition_;
  std::string name_;
};

StoreHandle::~StoreHandle() {}

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
  legate::ProvenanceTracker provenance(compiler->Name());
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto machine = core_runtime->get_machine();
  auto [start, stop] = compiler->MachineSlice();
  legate::MachineTracker tracker(machine.slice(start, stop));
  log_xla.debug() << "CreateCompileTask " << compiler->Name()
                  << " scheduling on slice [" << start << "," << stop << ")";
  size_t launch_size = compiler->LaunchSize();
  if ((stop - start) < launch_size) {
    std::stringstream sstr;
    sstr << "Not enough devices to run launch size " << launch_size
         << " on task " << compiler->Name();
    throw std::runtime_error(sstr.str());
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
    auto [start, stop] = compiler->MachineSlice();

    size_t launch_size = compiler->LaunchSize();
    legate::Shape flattened({launch_size});

    LOCK;
    auto runtime = legate_xla::Runtime::get_runtime();
    auto core_runtime = legate::Runtime::get_runtime();
    auto machine = core_runtime->get_machine();

    legate::ProvenanceTracker provenance(compiler->Name());
    legate::MachineTracker tracker(machine.slice(start, stop));
    log_xla.debug() << "CreateExecuteTask " << compiler->Name()
                    << " for launch on slice [" << start << "," << stop << ")";
    if (log_xla.want_debug()) {
      size_t input_idx = 0;
      for (const auto &input : inputs) {
        log_xla.debug() << compiler->Name() << " has input " << input_idx++
                        << ", name=" << input.impl->name()
                        << " with shape=" << input.impl->shape()
                        << ", store=" << input.impl.get()
                        << ", partitioned=" << std::boolalpha
                        << input.impl->HasPartition();
      }
      for (const auto &output : outputs) {
        log_xla.debug() << compiler->Name() << " has output "
                        << output.impl->name() << " with shape "
                        << output.impl->shape()
                        << ", store=" << output.impl.get()
                        << ", partitioned=" << std::boolalpha
                        << output.impl->HasPartition();
      }
    }
    if ((stop - start) < launch_size) {
      std::stringstream sstr;
      sstr << "Not enough devices to run launch shape " << launch_size
           << " on task " << compiler->Name() << std::endl;
      throw std::runtime_error(sstr.str());
    }
    auto task = runtime->create_task(XlaOpCode::XLA_EXECUTE_TASK, flattened);

    task.add_scalar_arg(
        legate::Scalar(reinterpret_cast<uint64_t>(compiler_hold)));
    task.add_scalar_arg(legate::Scalar(GetRunId()));

    task.add_scalar_arg(
        legate::Scalar(reinterpret_cast<uint64_t>(std::move(on_done))));

    auto check_replication_error = [=](const StoreHandle &store) {
      if (store.impl->HasPartition() && store.impl->shape().replicated > 1) {
        size_t tensor_launch_size =
            store.impl->shape().replicated * store.impl->shape().num_tiles;
        if (tensor_launch_size < launch_size) {
          // if this was created with a manual replication smaller than the
          // launch size then not all partitions will be satisfied
          std::stringstream sstr;
          sstr << "replicated tensor " << store.impl->name()
               << " cannot change replication from launch size "
               << tensor_launch_size << " to " << launch_size;
          throw std::runtime_error(sstr.str());
        }
      }
    };

    for (const auto &input : inputs) {
      check_replication_error(input);

      if (input.impl->HasPartition()) {
        task.add_input(input.impl->partition());
      } else {
        task.add_input(input.impl->store());
      }
    }

    for (const auto &output : outputs) {
      check_replication_error(output);

      if (output.impl->HasPartition()) {
        task.add_output(output.impl->partition());
      } else {
        task.add_output(output.impl->store());
      }
      task.add_scalar_arg(legate::Scalar(false));
    }
    if (launch_size > 1) {
      task.set_concurrent(true);
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
  size_t launch_size = LaunchSize(store.impl->shape(), num_local_devices);
  log_xla.debug() << "CopyDeviceToDevice with launch size " << launch_size;

  LOCK;
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto task =
      runtime->create_task(XlaOpCode::XLA_COPY_DEVICE_TO_DEVICE, {launch_size});
  if (store.impl->HasPartition()) {
    task.add_output(store.impl->partition());
  } else {
    task.add_output(store.impl->store());
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
  if (store.attached) {
    log_xla.debug() << "Detaching " << store.impl->name()
                    << ", store=" << store.impl.get();
    store.impl->store().detach();
  }
  log_xla.debug() << "Destroy array " << store.impl->name()
                  << ", store=" << store.impl.get();
  // no need to synchronize -- just removing the reference
  store.impl = nullptr;
}

void Synchronize(const StoreHandle &store) {
  LOCK;
  log_xla.debug() << "Synchronize store " << store.impl << " start";
  auto runtime = legate_xla::Runtime::get_runtime();
  auto &&logical_store = store.impl->store();
  auto out_mapped = logical_store.get_physical_store();
  auto buffer_alloc = legate::double_dispatch(
      out_mapped.dim(), out_mapped.code(), get_read_only_ptr{}, out_mapped);
  log_xla.debug() << "Synchronize store " << store.impl << " done";
}

void StoreBufferAction(const std::vector<BufferAction *> &actions,
                       const StoreHandle &store, BufferActionConfig config) {
  auto [start, stop] = config.machine_slice;
  size_t launch_size = LaunchSize(store.impl->shape(), stop - start);
  size_t num_local_devices = actions.size();
  log_xla.debug() << "legate_xla::StoreBufferAction with launch size "
                  << launch_size << " num_local_devices=" << num_local_devices;

  LOCK;
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto task =
      runtime->create_task(XlaOpCode::XLA_STORE_BUFFER_ACTION, {launch_size});

  auto machine = core_runtime->get_machine();

  log_xla.debug() << "legate_xla::StoreBufferAction: sliced onto [" << start
                  << "," << stop << ")";
  legate::MachineTracker tracker{machine.slice(start, stop)};

  TaskWaiter waiter(num_local_devices);
  task.add_scalar_arg(config.blocking);
  task.add_scalar_arg(reinterpret_cast<uint64_t>(&waiter));
  task.add_scalar_arg(num_local_devices);
  for (auto *action : actions) {
    task.add_scalar_arg(reinterpret_cast<uint64_t>(action));
  }

  if (store.impl->HasPartition()) {
    task.add_output(store.impl->partition());
  } else {
    task.add_output(store.impl->store());
  }
  runtime->submit(std::move(task));

  if (config.blocking) {
    waiter.Wait();
  }
}

bool IsMultiProcess(const legate::mapping::Machine &machine) {
  return (machine.processor_range().get_node_range().high - 1) !=
         machine.processor_range().get_node_range().low;
}

void SliceLocalShards(const StoreHandle &handle,
                      std::vector<void *> &local_shards,
                      std::pair<int64_t, int64_t> slice) {

  LOCK;
  auto [start, stop] = slice;

  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto machine = core_runtime->get_machine();

  std::optional<legate::MachineTracker> tracker;
  size_t launch_size = local_shards.size();
  if (IsMultiProcess(machine)) {
    launch_size =
        machine.processor_range().high - machine.processor_range().low;
    if (start != machine.processor_range().low ||
        stop != machine.processor_range().high) {
      throw std::runtime_error("SliceLocalShards: multi-process slice not run "
                               "on all nodes will violate control replication");
    }
  } else {
    launch_size = LaunchSize(handle.impl->shape(), local_shards.size());
    tracker.emplace(machine.slice(start, stop));
  }

  log_xla.debug() << "SliceLocalShards: slice " << local_shards.size()
                  << " shards on launch size " << launch_size << " on store "
                  << handle.impl.get() << " on machine slice [" << start << ","
                  << stop << ")";

  auto task =
      runtime->create_task(XlaOpCode::XLA_SHARD_GETTER_TASK, {launch_size});

  TaskWaiter waiter{int64_t(local_shards.size())};
  task.add_scalar_arg(reinterpret_cast<uint64_t>(local_shards.data()));
  task.add_scalar_arg(int64_t(local_shards.size()));
  task.add_scalar_arg(reinterpret_cast<uint64_t>(&waiter));

  if (handle.impl->HasPartition()) {
    task.add_input(handle.impl->partition());
  } else {
    task.add_input(handle.impl->store());
  }
  runtime->submit(std::move(task));

  waiter.Wait();
}

StoreHandle AssembleShards(const legate_xla::Shape &logical_shape,
                           const std::vector<legate_xla::Shard> &local_shards,
                           std::pair<int64_t, int64_t> slice) {
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
#if 0
  LOCK;
  auto shape = ComputeStoreShape(logical_shape);

  std::vector<std::pair<legate::ExternalAllocation, legate::Shape>> allocs;
  allocs.reserve(local_shards.size());
  for (const auto &shard : local_shards) {
    auto alloc = legate::ExternalAllocation::create_fbmem(
        shard.local_device_id, uintptr_t(shard.data), shard.size);
    allocs.emplace_back(std::move(alloc), legate::Shape(shard.shape_index));
  }

  auto legate_shape = legate::Shape{shape.dims};
  auto type = legate::primitive_type(SupportedTypeToLegateType(shape.type));
  auto legate_store = core_runtime->create_store(
      legate_shape, legate::Shape(shape.tile_shape), type, allocs);

  StoreHandle store{
      .impl = std::make_shared<StoreHandleImpl>(
          StoreHandleImpl{.store = std::move(legate_store.first),
                          .shape = shape,
                          .partition = std::move(legate_store.second),
                          .name = "dev_assembled"}),
      .attached = true,
  };
#else
  auto store = CreateStore(logical_shape, "assembled");
  auto [start, stop] = slice;
  log_xla.debug() << "AssembleShards: assembling " << local_shards.size()
                  << " local shards " << store.impl.get() << " on slice=["
                  << start << "," << stop << ")";
  uint64_t launch_size = stop - start;
  auto machine = core_runtime->get_machine();
  legate::MachineTracker tracker(machine.slice(start, stop));
  auto task =
      runtime->create_task(XlaOpCode::XLA_SHARD_ASSEMBLE_TASK, {launch_size});

  task.add_scalar_arg(int64_t(local_shards.size()));
  TaskWaiter waiter{int64_t(local_shards.size())};
  task.add_scalar_arg(reinterpret_cast<uint64_t>(&waiter));
  for (const auto &shard : local_shards) {
    task.add_scalar_arg(reinterpret_cast<int64_t>(shard.data));
  }

  // Treat this as an output for future synchronization purposes
  if (store.impl->HasPartition()) {
    task.add_output(store.impl->partition());
  } else {
    task.add_output(store.impl->store());
  }
  runtime->submit(std::move(task));
  waiter.Wait();
#endif
  return store;
}

StoreHandle Reshard(const StoreHandle &handle,
                    const std::vector<int64_t> &tile_shape) {
  if (!handle.impl->shape().tile_shape.has_value() ||
      tile_shape != *handle.impl->shape().tile_shape) {
    Shape new_shape = handle.impl->shape();
    new_shape.tile_shape = tile_shape;
    auto core_runtime = legate::Runtime::get_runtime();

    const auto &range = core_runtime->get_machine().processor_range();
    auto global_size = range.high - range.low;
    Shape store_shape = ComputeStoreShape(new_shape, global_size);
    auto new_impl = std::make_shared<StoreHandleImpl>(
        handle.impl->store(), std::move(new_shape), handle.impl->name());

    log_xla.debug() << "Reshard " << handle.impl.get()
                    << ", name=" << handle.impl->name()
                    << ", prev=" << handle.impl->shape()
                    << ", new=" << new_impl->shape()
                    << ", store=" << new_impl.get();

    std::vector<size_t> legate_tile_shape{tile_shape.begin(), tile_shape.end()};
    new_impl->SetPartition(
        new_impl->store().partition_by_tiling(legate_tile_shape));

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

  auto core_runtime = legate::Runtime::get_runtime();
  const auto &range = core_runtime->get_machine().processor_range();
  auto global_size = range.high - range.low;

  auto store_shape = ComputeStoreShape(shape, global_size);
  bool is_scalar = false;

  LOCK;
  std::string store_name = name.has_value() ? *name : "anonymous";
  StoreHandle result = {
      .impl = std::make_shared<StoreHandleImpl>(
          core_runtime->create_store(
              legate::Shape({store_shape.dims.begin(), store_shape.dims.end()}),
              legate::primitive_type(code), is_scalar),
          shape, std::move(store_name))};

  log_xla.debug() << "CreateStore " << result.impl.get()
                  << ", name=" << result.impl->name() << ", logical=" << shape
                  << ", actual=" << store_shape;

  if (store_shape.tile_shape.has_value()) {
    result.impl->SetPartition(result.impl->store().partition_by_tiling(
        {store_shape.tile_shape->begin(), store_shape.tile_shape->end()}));
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

  task.add_scalar_arg(scalar);
  if (handle.impl->HasPartition()) {
    task.add_output(handle.impl->partition());
  } else {
    task.add_output(handle.impl->store());
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
    log_xla.info() << "Legate stopped!";
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
  const auto &tile =
      shape.tile_shape.has_value() ? *shape.tile_shape : shape.dims;
  os << "Shape(type="
     << static_cast<std::underlying_type<SupportedType>::type>(shape.type)
     << ",dim=" << shape.dims << "),tile=" << tile
     << ", replication=" << shape.replicated;
  return os;
}

void initialize_runtime_and_context(legate::Runtime *runtime,
                                    legate::Library context) {

  legate_xla::Runtime::initialize(runtime, context);
}

} // namespace legate_xla

extern "C" void LegateShutdown() {
  // Make sure to clear all handles held by Legate
  // so that nothing gets deleted during program cleanup
  legate_xla::log_xla.info() << "Shutting down Legate";
  legate_xla::StopLegate();
  ShutdownLegateClient();
}
