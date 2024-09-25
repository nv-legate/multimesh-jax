#include "legate_to_xla.h"
#include "legate_xla_c.h"
#include "legate_xla_common.h"
#include "xla_to_legate.h"

#include <zuku/store.h>
#include <zuku/future.h>
#include <zuku/defer.h>
#include <zuku/tiled_array.h>

#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <atomic>
#include <realm.h>

#include "hlo_executor.h"
#include "legate_runtime.h"


using zuku::defer;
using zuku::on;
using zuku::after;

namespace legate_xla {

struct StoreHandleImpl {
  std::string name;

  zuku::Store<zuku::ShardedArray> array;
};

StoreHandle::~StoreHandle() {}


namespace {


static std::unordered_map</*processor=*/int64_t, Realm::Event> last_execute_task;

static std::set<Realm::Event> pending_compilation_events;

int64_t GetRunId() {
  static std::atomic<int64_t> counter{0};
  return counter.fetch_add(1);
}

std::vector<int64_t> LocalDeviceIds(const zuku::DeviceList& devices){
  return {};
}

Realm::Processor LocalProcessor(int64_t local_index){
  return Realm::Processor{};
}

std::vector<Realm::Processor> LocalProcessors(const std::pair<int64_t, int64_t>& slice)
{
  // TODO: return the actual processors
  return {};
}

Realm::Event LastExecuteTask(const Realm::Processor& p){
  //TODO: actually look this up
  return Realm::Event::NO_EVENT;
}

void SetLastExecuteTask(const Realm::Processor& p, Realm::Event ev){
  //TODO: set last event
}

int64_t IsLocal(const std::pair<int64_t,int64_t> machine_slice){
  // TODO; fix device index check
  return true;
}

zuku::View<zuku::ShardedArray> GetView(const StoreHandle& handle){
  return handle.impl->array.view();
}

zuku::Store<zuku::ShardedArray> GetStore(const StoreHandle& handle){
  return handle.impl->array.child();
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

void CreateCompileTask(std::shared_ptr<LegateCompiler> compiler) {
  // TODO: put this on the correct processor
  auto [start, stop] = compiler->MachineSlice();

  auto local_procs = LocalProcessors(compiler->MachineSlice());
  if (local_procs.empty()){
    return;
  }

  // TODO, rotate which GPUs are used for compilation
  auto token = zuku::on(local_procs[0]).defer([](int64_t run_id, std::shared_ptr<LegateCompiler> compiler){
    // TODO: execute the compiler task
  }, GetRunId(), std::move(compiler));

  // TODO: fix device index
  const int64_t device_index = 0;
  if (device_index < start || device_index >= stop){
    return;
  }

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
                       std::shared_ptr<LegateCompiler> compiler,
                       const std::vector<ScalarArgument> &scalars,
                       const std::vector<StoreHandle> &inputs,
                       const std::vector<StoreHandle> &outputs,
                       std::vector<std::function<void()>> *on_done) {
  
  Realm::Processor p = LocalProcessor(local_device_id);

  std::vector<Realm::Processor> local_procs = LocalProcessors(machine_slice);
  int64_t index = 0;
  Realm::Event prev_task = LastExecuteTask(p);
  auto view_inputs = GetViews(index, inputs);
  auto store_outputs = GetStores(index, outputs);
  auto token = zuku::on(p).after(prev_task).defer([](int64_t run_id, std::shared_ptr<LegateCompiler> compiler, std::vector<ScalarArgument> scalars, zuku::view_vector<ShardedArray> inputs, zuku::store_vector<ShardedArray> outputs){
    // TODO: execute the task
  }, run_id, compiler, scalars, std::move(view_inputs), std::move(store_outputs));

  if (on_done){
    after(token).defer([](std::function<void()> callback){
      callback();
    }, on_done->at(index));
  }
  SetLastExecuteTask(p, token.Event());
}

void Destroy(StoreHandle &store) {
  LOCK;

  log_xla.debug() << "Destroy array " << store.impl->name
                  << ", store=" << store.impl.get();
  // no need to synchronize -- just removing the reference
  store.impl = nullptr;
}

void StoreBufferAction(int64_t local_device_id, BufferAction* actions,
                       const StoreHandle &store, BufferActionConfig config) {
  Realm::Processor p = LocalProcessor(local_device_id);
  auto token = on(p).defer([](BufferAction* action, zuku::ShardedArray& array){
    // TODO: execute the buffer action
  }, action, GetStore(store));

  if (config.blocking){
    token.Wait();
  }
}

void SliceLocalShard(int64_t local_device_id, const StoreHandle &handle, void* host_buffer){
  Realm::Processor p = LocalProcessor(local_device_id);
  auto token = on(p).defer([](void* host_buffer, zuku::ShardedArray& array){
      // do the slicing
    }, host_buffer, GetStore(handle));
  token.Wait();
}

StoreHandle AssembleShards(int64_t local_device_id, zuku::ShardedShape shape,
                    const legate_xla::Shard &local_shard,
                    std::shared_ptr<LegateStream> stream,
                    std::optional<StoreHandle> existing_store) {
  Realm::Processor p = LocalProcessor(local_device_id);

  auto token = on(p).defer([](legate_xla::Shard shard, zuku::ShardedArray& array){
    // do the slicing
  }, shard, GetStore(store));

  if (!IsGpu()){
    token.Wait();
  }

  return store;
}

StoreHandle Reshard(const StoreHandle &handle, zuku::ShardedShape reshard_shape) {
  // TODO implement a resharding plan for this and do the exchange

  return StoreHandle{};
}

bool IsGpu() {
  // TODO: fix this
  return false;
}

void StartTimer(const std::string &name) {
  // TODO: start a timer
}

void StopTimer(const std::string &name) {
  // TODO: stop a timer
}

StoreHandle CreateStore(zuku::ShardedShape shape,
                        std::optional<std::string> name) {
  auto local_devices = LocalDeviceIds(shape.sharding.mesh.devices);
  std::vector<StoreHandleImpl::ArrayVariant> arrays;
  arrays.reserve(local_devices.size());
  for (int64_t dev : local_devices){
    auto store = zuku::ShardedArray::Create(std::move(shape));
    arrays.push_back(std::move(store));
  }

  std::string array_name = [&]{
    if (name.has_value()){
      return *std::move(name);
    }
    return std::string("anonymous");
  }();

  StoreHandleImpl impl{ .name = std::move(array_name), .arrays = std::move(arrays) };
  return StoreHandle{
    .impl = std::make_shared<StoreHandleImpl>(std::move(impl)),
    .unique_id = NextStoreId()
  };
}

void FenceCompilation() {
  Realm::Event all_compilations = Realm::Event::merge_events(pending_compilation_events);
  pending_compilation_events.clear();
  all_compilations.wait();
}

void StopLegate() {
  // TODO: stop the Realm runtime
}

void StartLegate() {
  // TODO: initialize the Realm runtime
}

} // namespace legate_xla

extern "C" void LegateShutdown() {
  // Make sure to clear all handles held by Legate
  // so that nothing gets deleted during program cleanup
  legate_xla::log_xla.info() << "Shutting down Legate";
  legate_xla::StopLegate();
  ShutdownLegateClient();
}
