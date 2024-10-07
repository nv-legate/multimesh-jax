#include "legate_to_xla.h"
#include "legate_xla_common.h"
#include "xla_task.h"
#include "xla_to_legate.h"

#include <mesh.h>
#include <processor.h>
#include <realm/memory.h>
#include <realm/processor.h>
#include <shape.h>
#include <zuku/defer.h>
#include <zuku/future.h>
#include <zuku/reshard.h>
#include <zuku/store.h>
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

namespace legate_xla {

struct StoreHandleImpl {
  std::string name;

  zuku::Store<zuku::ShardedArray> array;
};

StoreHandle::~StoreHandle() {}

struct BufferHandleImpl {
  zuku::Future<zuku::ArrayTile> tile;
};

BufferHandle::~BufferHandle() {}

namespace {

StartupConfig startup_config;

Realm::Runtime rt;

std::vector<Realm::Event> last_execute_tasks;

std::set<Realm::Event> pending_compilation_events;

int64_t GetRunId() {
  static std::atomic<int64_t> counter{0};
  return counter.fetch_add(1);
}

zuku::Processor LocalProcessor(int64_t local_index) {
  return zuku::Processor::Create({.local = local_index});
}

Realm::Event LastExecuteTask(const zuku::Processor &p) {
  return last_execute_tasks[p.local_id()];
}

void SetLastExecuteTask(const zuku::Processor &p, Realm::Event ev) {
  last_execute_tasks[p.local_id()] =
      Realm::Event::merge_events(last_execute_tasks[p.local_id()], ev);
}

zuku::View<zuku::ShardedArray> GetView(const StoreHandle &handle) {
  return handle.impl->array.view();
}

std::vector<zuku::View<zuku::ShardedArray>>
GetViews(const std::vector<StoreHandle> &handles,
         const std::set<int64_t> &output_ids) {
  std::vector<zuku::View<zuku::ShardedArray>> views;
  views.reserve(handles.size());
  for (auto &&handle : handles) {
    if (output_ids.find(handle.unique_id) != output_ids.end()) {
      // create a view outside the dependency analysis
      // this will also be passed as an output
      views.push_back(handle.impl->array.unsafe_view());
    } else {
      views.push_back(GetView(handle));
    }
  }
  return views;
}

zuku::Store<zuku::ShardedArray> GetStore(const StoreHandle &handle) {
  return handle.impl->array.child();
}

std::vector<zuku::Store<zuku::ShardedArray>>
GetStores(const std::vector<StoreHandle> &handles) {
  std::vector<zuku::Store<zuku::ShardedArray>> stores;
  stores.reserve(handles.size());
  for (auto &&handle : handles) {
    stores.push_back(GetStore(handle));
  }
  return stores;
}

static int64_t NextStoreId() {
  static std::atomic<int64_t> next_id{0};
  return next_id.fetch_add(int64_t(1));
}

#if 0
void LocalFence(const std::vector<StoreHandle> &inputs,
                const std::vector<StoreHandle> &inouts,
                std::pair<int64_t, int64_t> device_slice, std::string name) {
  log_xla.debug() << "Fencing " << inputs.size() << " inputs and "
                  << inouts.size() << " outputs on slice ["
                  << device_slice.first << "..." << device_slice.second << ")";
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto machine = core_runtime->get_machine();
  const uint64_t launch_size = device_slice.second - device_slice.first;
  auto scope =
      legate::Scope("LocalFence" + name)
          .with_machine(machine.slice(device_slice.first, device_slice.second));

  auto task = runtime->create_task(XlaOpCode::XLA_FENCE_TASK,
                                   legate::Shape({launch_size}));

  for (const auto &store : inputs) {
    log_xla.debug() << "Fence input " << store.impl->name();
    if (store.impl->HasPartition()) {
      task.add_input(store.impl->partition());
    } else {
      task.add_input(store.impl->store());
    }
  }
  for (const auto &store : inouts) {
    log_xla.debug() << "Fence in/out " << store.impl->name();
    if (store.impl->HasPartition()) {
      task.add_input(store.impl->partition());
      task.add_output(store.impl->partition());
    } else {
      task.add_input(store.impl->store());
      task.add_output(store.impl->store());
    }
  }
  runtime->submit(std::move(task));
}
#endif

#if 0
void OffloadDtoH(const std::vector<StoreHandle> &to_offload,
                 std::pair<int64_t, int64_t> device_slice, std::string name,
                 bool save_values) {
  log_xla.debug() << "Offloading " << to_offload.size() << " stores from task "
                  << name << " on slice [" << device_slice.first << "..."
                  << device_slice.second << ")";
  auto runtime = legate_xla::Runtime::get_runtime();
  auto core_runtime = legate::Runtime::get_runtime();
  auto machine = core_runtime->get_machine();
  const uint64_t launch_size = device_slice.second - device_slice.first;
  auto scope =
      legate::Scope("OffloadDtoH" + std::move(name))
          .with_machine(machine.only(legate::mapping::TaskTarget::CPU)
                            .slice(device_slice.first, device_slice.second));

  auto task = runtime->create_task(XlaOpCode::XLA_OFFLOAD_TASK,
                                   legate::Shape({launch_size}));

  for (const auto &store : to_offload) {
    log_xla.debug() << "OffloadDtoH" << store.impl->name();
    if (store.impl->HasPartition()) {
      if (save_values) {
        task.add_input(store.impl->partition());
      }
      task.add_output(store.impl->partition());
    } else {
      if (save_values) {
        task.add_input(store.impl->store());
      }
      task.add_output(store.impl->store());
    }
  }
  runtime->submit(std::move(task));
  core_runtime->issue_mapping_fence();
}
#endif

} // namespace

std::set<int> GetLocalDevices() {
  std::set<int> procs;
  for (auto &&p : zuku::Processor::DefaultProcs()) {
    procs.insert(p.global_id());
  }
  return procs;
}

void CreateCompileTask(int64_t local_device_id,
                       std::shared_ptr<LegateCompiler> compiler) {
  zuku::Processor p = LocalProcessor(local_device_id);

  // TODO, rotate which GPUs are used for compilation
  auto token =
      zuku::across(compiler->MachineSlice())
          .if_on(p)
          .defer(
              [=](int64_t run_id, std::shared_ptr<LegateCompiler> compiler) {
                LoadAndCompile(/*run_id=*/0, p, compiler);
              },
              GetRunId(), std::move(compiler));

  pending_compilation_events.insert(token.Event());
}

#if 0
void OffloadDtoH(const std::vector<StoreHandle> &blocking_users,
                 const std::vector<StoreHandle> &to_offload,
                 std::pair<int64_t, int64_t> device_slice, std::string name) {

  // if there are blocking users, we want to ensure that all copies of the user
  // have been made before starting any offload tasks we achieve this by doing a
  // local "fence" which does a dummy read/write on the blocking users while
  // "reading" the stores to be offloaded
  if (!blocking_users.empty()) {
    LocalFence(to_offload, blocking_users, device_slice, name);
  }
  OffloadDtoH(to_offload, device_slice, std::move(name), /*save_values=*/true);
}

void InvalidateDeviceInstances(const std::vector<StoreHandle> &stores,
                               std::pair<int64_t, int64_t> device_slice,
                               std::string name) {
  OffloadDtoH(stores, device_slice, std::move(name), /*save_values=*/false);
}
#endif

void CreateExecuteTask(int64_t run_id, int64_t local_device_id,
                       zuku::DeviceList devices,
                       std::shared_ptr<LegateCompiler> compiler,
                       const std::vector<ScalarArgument> &scalars,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       const BufferHandle &temp_buffer,
                       std::vector<std::function<void()>> *on_done) {
  zuku::Processor p = LocalProcessor(local_device_id);

  // if any of the outputs overlap with the inputs, then the inputs should be an
  // unsafe view
  std::set<int64_t> output_ids;
  for (auto &&output : outputs) {
    output_ids.insert(output.unique_id);
  }

  int64_t index = 0;
  Realm::Event prev_task = LastExecuteTask(p);
  auto view_inputs = GetViews(inputs, output_ids);
  auto store_outputs = GetStores(outputs);
  zuku::View<zuku::ArrayTile> temp = temp_buffer.impl->tile.view();

  auto token =
      zuku::across(std::move(devices))
          .if_on(p)
          .after(prev_task)
          .defer(
              [=](int64_t run_id, std::shared_ptr<LegateCompiler> compiler,
                  std::vector<ScalarArgument> scalars,
                  zuku::ro_vector<zuku::ShardedArray> inputs,
                  zuku::rw_vector<zuku::ShardedArray> outputs,
                  const zuku::ArrayTile &temp) {
                RunExecutable(run_id, std::move(devices), std::move(p),
                              std::move(compiler), std::move(scalars),
                              std::move(inputs), std::move(outputs), temp);
              },
              run_id, compiler, scalars, std::move(view_inputs),
              std::move(store_outputs), std::move(temp));

  if (on_done) {
    after(token).defer([](std::function<void()> callback) { callback(); },
                       on_done->at(index));
  }
  SetLastExecuteTask(p, token.Event());
}

void Destroy(StoreHandle &store) {
  // no need to synchronize -- just removing the reference
  store.impl = nullptr;
}

void StoreBufferAction(int64_t local_device_id, BufferAction *action,
                       const StoreHandle &store, bool blocking) {
  zuku::Processor p = LocalProcessor(local_device_id);
  auto token = on(p).defer(
      [](int64_t local_device_id, BufferAction *action,
         zuku::ShardedArray &array) {
        ApplyStoreBufferAction(local_device_id, action, array);
      },
      local_device_id, action, GetStore(store));

  if (blocking) {
    token.Wait();
  }
}

void *SliceLocalShard(int64_t local_device_id, const StoreHandle &handle) {
  zuku::Processor p = LocalProcessor(local_device_id);
  auto [buffer] = on(p).defer(
      [](const zuku::ShardedArray &array) {
        // do the slicing
        return array.tile().data();
      },
      GetView(handle));
  return const_cast<void *>(buffer.wait_and_get());
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

  auto token = on(p).defer(
      [](std::shared_ptr<LegateStream> stream, zuku::Processor p,
         legate_xla::Shard shard, zuku::ShardedArray &array) {
        // TODO: do the slicing
        if (p.type() == zuku::Processor::Type::CPU) {
          ::memcpy(array.tile().data(), shard.data, shard.size);
        } else if (p.type() == zuku::Processor::Type::GPU) {
          stream->MemcpyDtoDAsync(array.tile().data(), shard.data, shard.size,
                                  /*cpu=*/false, shard.local_device_id);
        }
      },
      std::move(stream), p, shard, GetStore(output));

  if (!IsGpu()) {
    token.Wait();
  }

  return output;
}

void Reshard(int64_t local_device_id, int64_t global_device_id,
             const StoreHandle &src, const StoreHandle &dst) {
  zuku::View<zuku::ShardedArray> input = GetView(src);
  zuku::Store<zuku::ShardedArray> output = GetStore(dst);

  zuku::Processor p = LocalProcessor(local_device_id);

  // zuku will create and execute a resharding plan
  zuku::Reshard(std::move(p), std::move(input), std::move(output));
}

bool IsGpu() {
  return zuku::Processor::DefaultType() == zuku::Processor::Type::GPU;
}

void StartTimer(const std::string &name) {
  // TODO: start a timer
}

void StopTimer(const std::string &name) {
  // TODO: stop a timer
}

BufferHandle CreateBuffer(int64_t local_device_id, int64_t global_device_id,
                          int64_t size) {
  zuku::TileShape shape{
      .type = zuku::SupportedType::S8,
      .dims = {size},
  };
  zuku::Processor p = LocalProcessor(local_device_id);
  BufferHandleImpl impl{
      .tile = zuku::ArrayTile::CreateFuture(std::move(shape),
                                            {.processor = std::move(p)}),
  };
  return BufferHandle{.impl =
                          std::make_shared<BufferHandleImpl>(std::move(impl))};
}

StoreHandle CreateStore(int64_t local_device_id, int64_t global_device_id,
                        zuku::ShardedShape shape,
                        std::optional<std::string> name) {
  const zuku::DeviceList &devices = shape.sharding.devices;

  zuku::Processor p = LocalProcessor(local_device_id);
  zuku::Store<zuku::ShardedArray> array =
      zuku::ShardedArray::Create(std::move(shape), {.processor = std::move(p)});

  std::string array_name = [&] {
    if (name.has_value()) {
      return *std::move(name);
    }
    return std::string("anonymous");
  }();

  StoreHandleImpl impl{.name = std::move(array_name),
                       .array = std::move(array)};
  return StoreHandle{.impl = std::make_shared<StoreHandleImpl>(std::move(impl)),
                     .unique_id = NextStoreId()};
}

void FenceCompilation() {
  Realm::Event all_compilations =
      Realm::Event::merge_events(pending_compilation_events);
  pending_compilation_events.clear();
  all_compilations.wait();
}

void StopLegate() {
  static std::atomic<bool> stopped{false};
  bool is_not_stopped = false;
  bool is_stopped = true;
  if (stopped.compare_exchange_strong(is_not_stopped, is_stopped)) {
    Realm::Event last_event = Realm::Event::merge_events(last_execute_tasks);
    // explicitly wait for everything to finish
    last_event.wait();
    ShutdownLegateClient();
    zuku::stop(rt, last_event);
  }
}

void StartLegate() {
  std::array argv = {"legate-jax"};
  rt = zuku::Init(argv.size(), (char **)argv.data(),
                  {
                      .cpus = startup_config.cpus,
                      .gpus = startup_config.gpus,
                      .sysmem = startup_config.sysmem,
                      .fbmem = startup_config.fbmem,
                      .zcmem = startup_config.zcmem,
                      .network = startup_config.network.value_or("default"),
                      .kthreads = startup_config.kthreads,
                  });
  const auto &procs = zuku::Processor::DefaultProcs();
  last_execute_tasks.reserve(procs.size());
  for (auto &&proc : procs) {
    last_execute_tasks.push_back(Realm::Event::NO_EVENT);
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
