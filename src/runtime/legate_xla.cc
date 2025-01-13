#include "legate_to_xla.h"
#include "legate_xla_common.h"
#include "xla_task.h"
#include "xla_to_legate.h"

#include <chrono>
#include <mesh.h>
#include <processor.h>
#include <realm/event.h>
#include <realm/logging.h>
#include <realm/memory.h>
#include <realm/processor.h>
#include <shape.h>
#include <type_traits.h>
#include <zuku/defer.h>
#include <zuku/future.h>
#include <zuku/reshard.h>
#include <zuku/store.h>
#include <zuku/stream.h>
#include <zuku/tiled_array.h>

#include <atomic>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <realm.h>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>

#include "hlo_executor.h"
#include "hlo_loader.h"

using zuku::after;
using zuku::on;

namespace {

std::vector<zuku::Store<zuku::ShardedArray>> *keep_from_deleting{nullptr};
std::vector<zuku::Future<zuku::ArrayTile>> *keep_tiles_from_deleting{nullptr};
std::atomic<bool> runtime_stopped{false};

using std_timer = decltype(std::chrono::steady_clock::now());
std::unordered_map<std::string, zuku::Future<std_timer>> pending_timers;

legate_xla::StartupConfig startup_config;

Realm::Runtime rt;

std::vector<Realm::Event> last_execute_events;
std::vector<Realm::UserEvent> last_control_events;
std::vector<zuku::ArrayCache> host_caches;
std::vector<zuku::ArrayCache> device_caches;

std::set<Realm::Event> pending_compilation_events;

} // namespace

namespace legate_xla {

struct StoreHandleImpl {
  StoreHandleImpl(std::string n, zuku::Store<zuku::ShardedArray> &&a)
      : name(std::move(n)), array(std::move(a)) {}

  std::string name;

  zuku::Store<zuku::ShardedArray> array;

  StoreHandleImpl(const StoreHandleImpl &) = delete;

  ~StoreHandleImpl() {
    if (runtime_stopped.load()) {
      // this must be occurring during python shutdown
      // there is no more runtime anymore, which means
      // we need to "leak" the array
      keep_from_deleting->push_back(std::move(array));
    }
  }
};

StoreHandle::~StoreHandle() {}

struct BufferHandleImpl {
  BufferHandleImpl(zuku::Future<zuku::ArrayTile> &&t) : tile(std::move(t)) {}

  zuku::Future<zuku::ArrayTile> tile;
  ~BufferHandleImpl() {
    if (runtime_stopped.load()) {
      // this is happening during Python shutdown and the
      // tile must be "leaked" since there is no more runtime
      keep_tiles_from_deleting->push_back(std::move(tile));
    }
  }
};

BufferHandle::~BufferHandle() {}

Realm::Logger log_xla("legate.xla");

namespace {

int64_t GetRunId() {
  static std::atomic<int64_t> counter{0};
  return counter.fetch_add(1);
}

zuku::Processor LocalProcessor(int64_t local_index) {
  return zuku::Processor::Create({.local = local_index});
}

void SetLastExecuteEvent(const zuku::Processor &p, Realm::Event ev) {
  last_execute_events[p.local_id()] = Realm::Event::merge_events(
      last_execute_events[p.local_id()], std::move(ev));
}

void SetLastControlEvent(const zuku::Processor &p, Realm::UserEvent ev) {
  if (ev != Realm::UserEvent::NO_USER_EVENT) {
    last_control_events[p.local_id()] = ev;
  }
}

zuku::store_variant_vector<zuku::ShardedArray>
GetStores(const std::vector<StoreHandle> &handles,
          const std::set<int64_t> &output_ids) {
  zuku::store_variant_vector<zuku::ShardedArray> views;
  views.reserve(handles.size());
  for (auto &&handle : handles) {
    if (output_ids.find(handle.unique_id) != output_ids.end()) {
      // create a view outside the dependency analysis
      // this will also be passed as an output
      views.push_back(handle.impl->array.unsafe_view());
    } else {
      views.push_back(handle.impl->array);
    }
  }
  return views;
}

zuku::store_vector<zuku::ShardedArray>
GetStores(const std::vector<StoreHandle> &handles) {
  zuku::store_vector<zuku::ShardedArray> stores;
  stores.reserve(handles.size());
  for (auto &&handle : handles) {
    stores.push_back(handle.impl->array);
  }
  return stores;
}

static int64_t NextStoreId() {
  static std::atomic<int64_t> next_id{0};
  return next_id.fetch_add(int64_t(1));
}

} // namespace

std::set<int> GetLocalDevices() {
  std::set<int> procs;
  for (auto &&p : zuku::Processor::DefaultProcs()) {
    procs.insert(p.local_id());
  }
  return procs;
}

void OffloadHtoD(int64_t local_device_id,
                 const std::vector<StoreHandle> &to_offload,
                 const std::string &task_name) {
  // there shouldn't be any competing H->D traffic so just run this ASAP
  log_xla.debug() << "OffloadHtoD " << to_offload.size() << " stores for task "
                  << task_name << " on device " << local_device_id;

  for (auto &handle : to_offload) {
    // only move back if the task processor does not match the current array
    // processor
    if (handle.impl->array->processor().type() !=
        LocalProcessor(local_device_id).type()) {
      handle.impl->array = zuku::ShardedArray::MoveToMemory(
          std::move(handle.impl->array), host_caches[local_device_id],
          device_caches[local_device_id]);
    }
  }
}

void OffloadDtoH(int64_t local_device_id,
                 const std::vector<StoreHandle> &to_offload,
                 const std::vector<StoreHandle> &pipelined,
                 const std::string &task_name) {

  log_xla.debug() << "OffloadDtoH " << to_offload.size() << " stores from task "
                  << task_name << " on device " << local_device_id;

  // don't offload until all the pipelined intermediates have been sent
  // otherwise you will end up with serious PCI contention
  const Realm::Event precondition = [&] {
    if (pipelined.empty()) {
      return Realm::Event::NO_EVENT;
    }
    if (pipelined.size() == 1) {
      return pipelined.front().impl->array.Precondition();
    }
    std::set<Realm::Event> preconditions;
    for (auto &handle : pipelined) {
      preconditions.insert(handle.impl->array.Precondition());
    }
    return Realm::Event::merge_events(preconditions);
  }();

  for (auto &handle : to_offload) {
    // only offload if on the GPU
    if (handle.impl->array->processor().type() == zuku::Processor::Type::GPU) {
      handle.impl->array = zuku::ShardedArray::MoveToMemory(
          std::move(handle.impl->array), device_caches[local_device_id],
          host_caches[local_device_id], precondition);
    }
  }
}

void CreateCompileTask(int64_t local_device_id,
                       std::shared_ptr<LegateCompiler> compiler) {
  zuku::Processor p = LocalProcessor(local_device_id);

  log_xla.debug() << "CreateCompileTask: " << compiler->Name()
                  << " on local device " << local_device_id;
  auto token =
      zuku::across(compiler->MachineSlice())
          .region("Compile " + compiler->Name())
          .if_on(p)
          .defer(
              [=](int64_t run_id, std::shared_ptr<LegateCompiler> compiler) {
                LoadAndCompile(/*run_id=*/0, p, compiler);
              },
              GetRunId(), std::move(compiler));

  pending_compilation_events.insert(token.Event());
}

void CreateExecuteTask(int64_t run_id, int64_t local_device_id,
                       int64_t global_device_id, zuku::DeviceList devices,
                       std::shared_ptr<LegateCompiler> compiler,
                       const std::vector<ScalarArgument> &scalars,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       const BufferHandle &temp_buffer,
                       ExecuteOptions options) {
  zuku::Processor p = LocalProcessor(local_device_id);

  // if any of the outputs overlap with the inputs, then the inputs should be an
  // unsafe view
  std::set<int64_t> output_ids;
  for (auto &&output : outputs) {
    output_ids.insert(output.unique_id);
  }

  auto store_inputs = GetStores(inputs, output_ids);
  auto store_outputs = GetStores(outputs);
  zuku::View<zuku::ArrayTile> temp = temp_buffer.impl->tile.view();

  log_xla.debug() << "creating execute task " << compiler->Name()
                  << " across devices " << devices
                  << ", static_order=" << options.strict_ordering;

  if (log_xla.want_debug()) {
    for (auto &&input : inputs) {
      log_xla.debug() << compiler->Name() << " has input " << input.impl->name
                      << ", " << input.impl->array->shape()
                      << " with precondition "
                      << input.impl->array.Precondition();
    }

    for (auto &&output : outputs) {
      log_xla.debug() << compiler->Name() << " has output " << output.impl->name
                      << ", " << output.impl->array->shape()
                      << " with precondition "
                      << output.impl->array.Precondition();
    }
  }

  if (devices.Contains(global_device_id)) {
    for (auto &&output : outputs) {
      if (!output.impl->array->HasTile()) {
        std::cerr << "ouput " << output.impl->name << " has no tile on "
                  << global_device_id
                  << ", tensor_id=" << output.impl->array->mesh_unique_id()
                  << std::endl;
        abort();
      }
    }
    for (auto &&input : inputs) {
      if (!input.impl->array->HasTile()) {
        std::cerr << "input " << input.impl->name << " has no tile on "
                  << global_device_id
                  << ", tensor_id=" << input.impl->array->mesh_unique_id()
                  << std::endl;
        abort();
      }
    }
  }

  std::string profile_name = options.name.value_or(compiler->Name()) +
                             " processor " + std::to_string(p.global_id());

  const Realm::Event order_event = [&] {
    if (options.strict_ordering) {
      return last_control_events[p.local_id()];
    }
    return Realm::UserEvent::NO_USER_EVENT;
  }();

  auto token =
      zuku::across(devices)
          .if_on(p)
          .after(order_event)
          .region(profile_name)
          .priority(options.priority.value_or(0))
          .stream_ordered()
          .defer(
              [](zuku::Stream *zs, int64_t run_id, zuku::Processor p,
                 zuku::DeviceList devices,
                 std::shared_ptr<LegateCompiler> compiler,
                 std::vector<ScalarArgument> scalars,
                 zuku::ro_vector<zuku::ShardedArray> inputs,
                 zuku::rw_vector<zuku::ShardedArray> outputs,
                 const zuku::ArrayTile &temp) {
                RunExecutable(zs, run_id, std::move(devices), std::move(p),
                              std::move(compiler), std::move(scalars),
                              std::move(inputs), std::move(outputs), temp);
              },
              run_id, p, std::move(devices), compiler, scalars,
              std::move(store_inputs), std::move(store_outputs),
              std::move(temp));

  if (log_xla.want_debug()) {
    for (auto &&input : inputs) {
      log_xla.debug() << compiler->Name() << " has input " << input.impl->name
                      << ", " << input.impl->array->shape()
                      << " with postcondition "
                      << input.impl->array.Precondition();
    }

    for (auto &&output : outputs) {
      log_xla.debug() << compiler->Name() << " has output " << output.impl->name
                      << ", " << output.impl->array->shape()
                      << " with postcondition "
                      << output.impl->array.Precondition();
    }
  }

  SetLastExecuteEvent(p, token.Event());
  if (options.strict_ordering) {
    SetLastControlEvent(p, token.ControlEvent());
  }
}

void RunAfterAllTasks(int64_t local_device_id, std::function<void()> on_done) {
  if (on_done) {
    after(last_execute_events[local_device_id])
        .defer([](std::function<void()> callback) { callback(); },
               std::move(on_done));
  }
}

void Destroy(StoreHandle &store) {
  if (runtime_stopped.load() == false) {
    // no need to synchronize -- just removing the reference
    store.impl = nullptr;
  } // the runtime no longer exists, we can't delete
}

void StoreBufferAction(int64_t local_device_id, BufferAction *action,
                       const StoreHandle &store, bool blocking) {
  zuku::Processor p = LocalProcessor(local_device_id);
  auto token = on(p).defer(
      [](int64_t local_device_id, BufferAction *action,
         zuku::ShardedArray &array) {
        ApplyStoreBufferAction(local_device_id, action, array);
      },
      local_device_id, action, store.impl->array);

  if (blocking) {
    token.Wait();
  }
}

bool HasLocalShard(const legate_xla::StoreHandle &handle) {
  return handle.impl->array->HasTile();
}

void *SliceLocalShard(int64_t local_device_id, const StoreHandle &handle) {
  zuku::Processor p = LocalProcessor(local_device_id);
  auto [buffer] = on(p).defer(
      [](const zuku::ShardedArray &array) {
        // do the slicing
        return array.tile().data();
      },
      handle.impl->array);
  return const_cast<void *>(buffer.wait_and_get());
}

zuku::ShardedShape GetStoreShardedShape(const StoreHandle &handle) {
  return handle.impl->array->shape();
}

StoreHandle AssembleShards(int64_t local_device_id, int64_t global_device_id,
                           zuku::ShardedShape shape, legate_xla::Shard shard,
                           std::shared_ptr<LegateStream> stream,
                           std::optional<StoreHandle> existing_store) {
  zuku::Processor p = LocalProcessor(local_device_id);

  StoreHandle output = [&] {
    if (existing_store.has_value()) {
      return *std::move(existing_store);
    }
    return CreateStore(local_device_id, global_device_id, shape);
  }();

  log_xla.debug() << "AssembleShards: device=" << local_device_id
                  << ", shape=" << shape << ", name=" << output.impl->name
                  << ", id=" << output.unique_id;

  auto token =
      across(shape.sharding.devices)
          .if_on(p)
          .defer(
              [](std::shared_ptr<LegateStream> stream, zuku::Processor p,
                 legate_xla::Shard shard, zuku::ShardedArray &array) {
                // TODO: do the slicing
                if (p.type() == zuku::Processor::Type::CPU) {
                  ::memcpy(array.tile().data(), shard.data, shard.size);
                } else if (p.type() == zuku::Processor::Type::GPU) {
                  stream->MemcpyDtoDAsync(array.tile().data(), shard.data,
                                          shard.size,
                                          /*cpu=*/false, shard.local_device_id);
                }
              },
              std::move(stream), p, shard, output.impl->array);

  log_xla.debug() << (output.impl->array->HasName() ? output.impl->array->name()
                                                    : "anonymous")
                  << " has assemble postcondition "
                  << output.impl->array.Precondition();

  return output;
}

void Rename(StoreHandle &handle, std::string name) {
  log_xla.debug() << "renaming " << handle.impl->name << " to " << name
                  << ", tensor_id=" << handle.impl->array->mesh_unique_id();
  handle.impl->name = std::move(name);
}

void Reshard(int64_t local_device_id, int64_t global_device_id,
             const StoreHandle &src, const StoreHandle &dst) {
  zuku::Processor p = LocalProcessor(local_device_id);

  log_xla.debug() << "Reshard " << src.impl->name << " from "
                  << src.impl->array->shape() << " to "
                  << dst.impl->array->shape()
                  << " for source=" << src.impl->array->mesh_unique_id()
                  << " to dest=" << dst.impl->array->mesh_unique_id() << " on "
                  << global_device_id;

  // zuku will create and execute a resharding plan
  zuku::Reshard(std::move(p), src.impl->array, dst.impl->array);
}

bool IsGpu() {
  return zuku::Processor::DefaultType() == zuku::Processor::Type::GPU;
}

void StartTimer(const std::string &name) {
  auto [start_time] = after(last_execute_events[0]).defer([] {
    return std::chrono::steady_clock::now();
  });
  pending_timers.emplace(name, std::move(start_time));
}

void StopTimer(const std::string &name) {
  auto iter = pending_timers.find(name);
  if (iter == pending_timers.end()) {
    log_xla.warning() << "cannot stop timer " << name << ", does not exist";
    return;
  }

  after(last_execute_events[0])
      .defer(
          [](std::string name, std_timer start_timer) {
            auto stop_timer = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = stop_timer - start_timer;
            log_xla.info() << name << " finished in " << diff.count() << "s";
          },
          std::move(iter->first), std::move(iter->second));

  pending_timers.erase(iter);
}

BufferHandle CreateBuffer(int64_t local_device_id, int64_t global_device_id,
                          int64_t size) {
  Realm::Rect<1, long long> bounds;
  bounds.lo[0] = 0;
  bounds.hi[0] = std::max<int64_t>(0, size - 1);
  zuku::TileShape shape{.type = zuku::SupportedType::S8,
                        .dims = {size},
                        .bounds = std::move(bounds)};
  zuku::Processor p = LocalProcessor(local_device_id);
  auto tile = zuku::ArrayTile::CreateFuture(std::move(shape),
                                            {.processor = std::move(p)});
  return BufferHandle{.impl =
                          std::make_shared<BufferHandleImpl>(std::move(tile))};
}

void Free(int64_t local_device_id, legate_xla::StoreHandle handle) {
  auto shape = handle.impl->array->shape();
  log_xla.debug() << "FreeStore: device=" << local_device_id
                  << ", shape=" << shape;
  device_caches[local_device_id].Free(shape, std::move(handle.impl->array));
}

StoreHandle CreateStore(int64_t local_device_id, int64_t global_device_id,
                        zuku::ShardedShape shape,
                        std::optional<std::string> name,
                        std::optional<int64_t> min_cache_size) {
  log_xla.debug() << "CreateStore: device=" << local_device_id
                  << ", shape=" << shape
                  << ", name=" << name.value_or("anonymous");

  auto array =
      device_caches[local_device_id].Get(shape, std::move(min_cache_size));

  const int64_t next_id = NextStoreId();

  std::string array_name = [&] {
    if (name.has_value()) {
      return *std::move(name);
    }
    return std::string("anonymous");
  }();

  return StoreHandle{.impl = std::make_shared<StoreHandleImpl>(
                         std::move(array_name), std::move(array)),
                     .unique_id = next_id};
}

void FenceCompilation() {
  Realm::Event all_compilations =
      Realm::Event::merge_events(pending_compilation_events);
  pending_compilation_events.clear();
  all_compilations.wait();
}

void FenceExecution() {
  Realm::Event merged = Realm::Event::merge_events(last_execute_events);
  merged.wait();
}

void StopLegate() {

  bool is_not_stopped = false;
  bool is_stopped = true;
  if (runtime_stopped.compare_exchange_strong(is_not_stopped, is_stopped)) {
    // explicitly wait for everything to finish
    Realm::Event last_event = Realm::Event::merge_events(last_execute_events);
    last_event.wait();
    device_caches.clear();
    host_caches.clear();
    ShutdownLegateClient();
    zuku::stop(rt, last_event);
  }
}

void StartLegate() {
  std::array argv = {"legate-jax"};
  keep_from_deleting = new std::vector<zuku::Store<zuku::ShardedArray>>;
  keep_tiles_from_deleting = new std::vector<zuku::Future<zuku::ArrayTile>>;
  rt = zuku::Init(argv.size(), (char **)argv.data(),
                  {
                      .cpus = startup_config.cpus,
                      .gpus = startup_config.gpus,
                      .sysmem = startup_config.sysmem,
                      .fbmem = startup_config.fbmem,
                      .zcmem = startup_config.zcmem,
                      .network = startup_config.network.value_or("default"),
                      .kthreads = startup_config.kthreads,
                      .profile = startup_config.profile,
                      .argv = startup_config.argv,
                  });
  const auto &procs = zuku::Processor::DefaultProcs();
  zuku::Processor host = zuku::Processor::Util();
  last_execute_events.reserve(procs.size());
  last_control_events.reserve(procs.size());
  device_caches.reserve(procs.size());
  host_caches.reserve(procs.size());
  for (auto &&proc : procs) {
    last_execute_events.push_back(Realm::Event::NO_EVENT);
    last_control_events.push_back(Realm::UserEvent::NO_USER_EVENT);
    zuku::Processor host =
        zuku::Processor::Create(proc.id(), zuku::Processor::Type::CPU);
    host_caches.emplace_back(Realm::Memory::Kind::Z_COPY_MEM, host);
    device_caches.emplace_back(proc.DefaultMemoryKind(), proc);
  }

  // This has to come after PyFinalize
  atexit(StopLegate);
}

void SetStartupConfig(StartupConfig cfg) { startup_config = cfg; }

} // namespace legate_xla

extern "C" void SetStrictStaticOrder(bool flag) {}

extern "C" void LegateShutdown() {
  // Make sure to clear all handles held by Legate
  // so that nothing gets deleted during program cleanup
  legate_xla::StopLegate();
  ShutdownLegateClient();
}
