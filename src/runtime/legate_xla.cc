#include "legate_to_xla.h"
#include "legate_xla_common.h"
#include "xla_task.h"
#include "xla_to_legate.h"

#include <realm/memory.h>
#include <realm/processor.h>
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
#include "hlo_loader.h"


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
  //TODO: get the default local processor
  return Realm::Processor{};
}

Realm::Event LastExecuteTask(const Realm::Processor& p){
  //TODO: actually look this up
  return Realm::Event::NO_EVENT;
}

void SetLastExecuteTask(const Realm::Processor& p, Realm::Event ev){
  //TODO: set last event
}

zuku::View<zuku::ShardedArray> GetView(const StoreHandle& handle){
  return handle.impl->array.view();
}

std::vector<zuku::View<zuku::ShardedArray>> GetViews(const std::vector<StoreHandle>& handles){
  std::vector<zuku::View<zuku::ShardedArray>> views;
  views.reserve(handles.size());
  for (auto&& handle : handles){
    views.push_back(GetView(handle));
  }
  return views;
}

zuku::Store<zuku::ShardedArray> GetStore(const StoreHandle& handle){
  return handle.impl->array.child();
}

std::vector<zuku::Store<zuku::ShardedArray>> GetStores(const std::vector<StoreHandle>& handles){
  std::vector<zuku::Store<zuku::ShardedArray>> stores;
  stores.reserve(handles.size());
  for (auto&& handle : handles){
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

void CreateCompileTask(int64_t local_device_id, std::shared_ptr<LegateCompiler> compiler) {
  Realm::Processor p = LocalProcessor(local_device_id);

  // TODO, rotate which GPUs are used for compilation
  auto token = zuku::on(p).defer([](int64_t run_id, std::shared_ptr<LegateCompiler> compiler){
    // TODO: execute the compiler task
    LoadAndCompile(/*run_id=*/0, compiler);
  }, GetRunId(), std::move(compiler));

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
  int64_t index = 0;
  Realm::Event prev_task = LastExecuteTask(p);
  auto view_inputs = GetViews(inputs);
  auto store_outputs = GetStores(outputs);
  auto token = zuku::on(p).after(prev_task).defer([](int64_t run_id, std::shared_ptr<LegateCompiler> compiler, std::vector<ScalarArgument> scalars,
                                                             zuku::ro_vector<zuku::ShardedArray> inputs, zuku::rw_vector<zuku::ShardedArray> outputs){
    // TODO: execute the task
    RunExecutable(run_id, std::move(compiler), std::move(scalars), std::move(inputs), std::move(outputs));
  }, run_id, compiler, scalars, std::move(view_inputs), std::move(store_outputs));

  if (on_done){
    after(token).defer([](std::function<void()> callback){
      callback();
    }, on_done->at(index));
  }
  SetLastExecuteTask(p, token.Event());
}

void Destroy(StoreHandle &store) {
  // no need to synchronize -- just removing the reference
  store.impl = nullptr;
}

void StoreBufferAction(int64_t local_device_id, BufferAction* action,
                       const StoreHandle &store, bool blocking) {
  Realm::Processor p = LocalProcessor(local_device_id);
  auto token = on(p).defer([](int64_t local_device_id, BufferAction* action, zuku::ShardedArray& array){
    ApplyStoreBufferAction(local_device_id, action, array);
  }, local_device_id, action, GetStore(store));

  if (blocking){
    token.Wait();
  }
}

void* SliceLocalShard(int64_t local_device_id, const StoreHandle &handle){
  throw std::runtime_error("SliceLocalShard: unimplemented");
  Realm::Processor p = LocalProcessor(local_device_id);
  auto buffer = on(p).defer([](zuku::ShardedArray& array){
      // do the slicing
    }, GetStore(handle));
  return nullptr;
}

StoreHandle AssembleShards(int64_t local_device_id, zuku::ShardedShape shape,
                    const legate_xla::Shard &local_shard,
                    std::shared_ptr<LegateStream> stream,
                    std::optional<StoreHandle> existing_store) {
  Realm::Processor p = LocalProcessor(local_device_id);

  StoreHandle output = [&]{
    if (existing_store.has_value()){
      return *std::move(existing_store);
    }
    return CreateStore(local_device_id, shape);
  }();

  auto token = on(p).defer([](legate_xla::Shard shard, zuku::ShardedArray& array){
    //TODO: do the slicing
  }, local_shard, GetStore(output));

  if (!IsGpu()){
    token.Wait();
  }

  return output;
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

StoreHandle CreateStore(int64_t local_device_id, zuku::ShardedShape shape,
                        std::optional<std::string> name) {
  // TODO: select the default memory
  Realm::Memory m{};
  auto array = zuku::ShardedArray::Create(local_device_id, m, std::move(shape));
  std::string array_name = [&]{
    if (name.has_value()){
      return *std::move(name);
    }
    return std::string("anonymous");
  }();

  StoreHandleImpl impl{ .name = std::move(array_name), .array = std::move(array) };
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
  legate_xla::StopLegate();
  ShutdownLegateClient();
}
