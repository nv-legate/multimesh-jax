/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/zuku_execute_context_impl.h"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "hlo_executor.h"
#include "hlo_loader.h"
#include "realm.h"
#include "realm/logging.h"
#include "realm/memory.h"
#include "src/zuku/defer.h"
#include "src/zuku/future.h"
#include "src/zuku/mesh.h"
#include "src/zuku/processor.h"
#include "src/zuku/reshard.h"
#include "src/zuku/shape.h"
#include "src/zuku/store.h"
#include "src/zuku/stream.h"
#include "xla/pjrt/multimesh/mm_utils.h"
#include "xla/pjrt/multimesh/store_handle.h"

using zuku::after;
using zuku::on;

namespace {

std::vector<zuku::Store<zuku::ShardedArray>>* keep_from_deleting{nullptr};
std::vector<zuku::Future<zuku::ArrayTile>>* keep_tiles_from_deleting{nullptr};
std::atomic<bool> runtime_stopped{false};
std::optional<Realm::Runtime> rt;
Realm::Event sync_all_execution_event{Realm::Event::NO_EVENT};

constexpr int kMaxOpenWindows = 8;

void StopZuku() {
  bool is_not_stopped = false;
  bool is_stopped = true;
  if (rt.has_value() &&
      runtime_stopped.compare_exchange_strong(is_not_stopped, is_stopped)) {
    // explicitly wait for everything to finish
    sync_all_execution_event.wait();
    xla::ClearCachedStreams();
    zuku::stop(*rt, sync_all_execution_event);
  }
}

}  // namespace

namespace xla {

StoreHandleImpl::~StoreHandleImpl() {
  if (runtime_stopped.load()) {
    // this must be occurring during python shutdown
    // there is no more runtime anymore, which means
    // we need to "leak" the array
    keep_from_deleting->push_back(std::move(array));
  }
}

// Destructor cannot be defined in the header file due to PIMPL
// NOLINTNEXTLINE(modernize-use-equals-default)
StoreHandle::~StoreHandle() {}

Realm::Logger log_xla("multimesh.jax");

namespace {

int64_t GetRunId() {
  static std::atomic<int64_t> counter{0};
  return counter.fetch_add(1);
}

int64_t NextStoreId() {
  static std::atomic<int64_t> next_id{0};
  return next_id.fetch_add(int64_t(1));
}

}  // namespace

std::set<int> ZukuExecuteContextImpl::GetLocalDevices() {
  std::set<int> procs;
  for (auto&& p : zuku::Processor::DefaultProcs()) {
    procs.insert(p.local_id());
  }
  return procs;
}

zuku::Processor ZukuExecuteContextImpl::LocalProcessor(int64_t local_index) {
  return zuku::Processor::Create({.local = local_index});
}

void ZukuExecuteContextImpl::SetLastExecuteEvent(const zuku::Processor& p,
                                                 Realm::Event ev) {
  last_execute_events_[p.local_id()] = Realm::Event::merge_events(
      last_execute_events_[p.local_id()], std::move(ev));
}

void ZukuExecuteContextImpl::SetLastControlEvent(const zuku::Processor& p,
                                                 Realm::UserEvent ev) {
  if (ev != Realm::UserEvent::NO_USER_EVENT) {
    last_control_events_[p.local_id()] = ev;
  }
}

zuku::store_vector<zuku::ShardedArray> ZukuExecuteContextImpl::GetStores(
    const std::vector<StoreHandle>& handles) {
  zuku::store_vector<zuku::ShardedArray> stores;
  stores.reserve(handles.size());
  for (auto&& handle : handles) {
    stores.push_back(handle.impl->array);
  }
  return stores;
}

void ZukuExecuteContextImpl::OpenWindow() {
  // we have to limit the maximum number of open windows
  // to make sure the Realm event graph doesn't explode in size
  while (window_markers_.size() >= kMaxOpenWindows) {
    Realm::Event wait = window_markers_.front();
    window_markers_.pop_front();
    wait.wait();
  }
}

void ZukuExecuteContextImpl::CloseWindow() {
  Realm::Event marker = Realm::Event::merge_events(last_execute_events_);
  // window_markers_.push_back(marker);
  sync_all_execution_event =
      Realm::Event::merge_events(sync_all_execution_event, marker);
}

zuku::store_variant_vector<zuku::ShardedArray>
ZukuExecuteContextImpl::GetStores(const std::vector<StoreHandle>& handles,
                                  const std::set<int64_t>& output_ids) {
  zuku::store_variant_vector<zuku::ShardedArray> views;
  views.reserve(handles.size());
  for (auto&& handle : handles) {
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

void ZukuExecuteContextImpl::OffloadHtoD(
    int64_t local_device_id, const std::vector<StoreHandle>& to_offload,
    const std::string& task_name) {
  // there shouldn't be any competing H->D traffic so just run this ASAP
  log_xla.debug() << "OffloadHtoD " << to_offload.size() << " stores for task "
                  << task_name << " on device " << local_device_id;

  for (auto& handle : to_offload) {
    // only move back if the task processor does not match the current array
    // processor
    if (handle.impl->array->processor().type() !=
        LocalProcessor(local_device_id).type()) {
      handle.impl->array = zuku::ShardedArray::MoveToMemory(
          std::move(handle.impl->array), host_caches_[local_device_id],
          device_caches_[local_device_id]);
    }
  }
}

void ZukuExecuteContextImpl::OffloadDtoH(
    int64_t local_device_id, const std::vector<StoreHandle>& to_offload,
    const std::vector<StoreHandle>& pipelined, const std::string& task_name) {
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
    for (auto& handle : pipelined) {
      preconditions.insert(handle.impl->array.Precondition());
    }
    return Realm::Event::merge_events(preconditions);
  }();

  for (auto& handle : to_offload) {
    // only offload if on the GPU
    if (handle.impl->array->processor().type() == zuku::Processor::Type::GPU) {
      handle.impl->array = zuku::ShardedArray::MoveToMemory(
          std::move(handle.impl->array), device_caches_[local_device_id],
          host_caches_[local_device_id], precondition);
    }
  }
}

void ZukuExecuteContextImpl::CreateCompileTask(
    int64_t local_device_id, std::shared_ptr<MultiMeshCompiler> compiler) {
  zuku::Processor p = LocalProcessor(local_device_id);

  log_xla.debug() << "CreateCompileTask: " << compiler->Name()
                  << " on local device " << local_device_id;
  auto token =
      zuku::on(p)
          .region("Compile " + compiler->Name())
          .defer(
              [=](int64_t run_id, std::shared_ptr<MultiMeshCompiler> compiler) {
                LoadAndCompile(/*run_id=*/0, p, compiler);
              },
              GetRunId(), std::move(compiler));

  pending_compilation_events_.insert(token.Event());
}

void ZukuExecuteContextImpl::CreateExecuteTask(
    int64_t run_id, int64_t local_device_id, int64_t global_device_id,
    zuku::DeviceList devices, std::shared_ptr<MultiMeshCompiler> compiler,
    const std::vector<ScalarArgument>& scalars,
    const std::vector<StoreHandle>& inputs,
    const std::vector<StoreHandle>& outputs,
    zuku::Future<zuku::ArrayTile>& temp_buffer,
    MultiMeshExecuteOptions options) {
  zuku::Processor p = LocalProcessor(local_device_id);

  // if any of the outputs overlap with the inputs, then the inputs should be an
  // unsafe view
  std::set<int64_t> output_ids;
  for (auto&& output : outputs) {
    output_ids.insert(output.unique_id);
  }

  auto store_inputs = GetStores(inputs, output_ids);
  auto store_outputs = GetStores(outputs);

  log_xla.debug() << "creating execute task " << compiler->Name()
                  << " across devices " << devices
                  << ", static_order=" << options.strict_ordering;

  if (log_xla.want_debug()) {
    for (auto&& input : inputs) {
      log_xla.debug() << compiler->Name() << " has input " << input.impl->name
                      << ", " << input.impl->array->shape()
                      << " with precondition "
                      << input.impl->array.Precondition();
    }

    for (auto&& output : outputs) {
      log_xla.debug() << compiler->Name() << " has output " << output.impl->name
                      << ", " << output.impl->array->shape()
                      << " with precondition "
                      << output.impl->array.Precondition();
    }
  }

  if (devices.Contains(global_device_id)) {
    for (auto&& output : outputs) {
      if (!output.impl->array->HasTile()) {
        std::cerr << "output " << output.impl->name << " has no tile on "
                  << global_device_id
                  << ", tensor_id=" << output.impl->array->mesh_unique_id()
                  << " for task " << compiler->Name() << std::endl;
        abort();
      }
    }
    for (auto&& input : inputs) {
      if (!input.impl->array->HasTile()) {
        std::cerr << "input " << input.impl->name << " has no tile on "
                  << global_device_id
                  << ", tensor_id=" << input.impl->array->mesh_unique_id()
                  << " for task " << compiler->Name() << std::endl;
        abort();
      }
    }
  }

  std::string profile_name = options.name.value_or(compiler->Name()) +
                             " processor " + std::to_string(p.global_id());

  const Realm::Event order_event = [&] {
    if (options.strict_ordering) {
      return last_control_events_[p.local_id()];
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
              [](zuku::Stream* zs, int64_t run_id, zuku::Processor p,
                 zuku::DeviceList devices,
                 std::shared_ptr<MultiMeshCompiler> compiler,
                 std::vector<ScalarArgument> scalars,
                 zuku::ro_vector<zuku::ShardedArray> inputs,
                 zuku::rw_vector<zuku::ShardedArray> outputs,
                 const zuku::ArrayTile& temp) {
                RunExecutable(zs, run_id, std::move(devices), std::move(p),
                              std::move(compiler), std::move(scalars),
                              std::move(inputs), std::move(outputs), temp);
              },
              run_id, p, std::move(devices), compiler, scalars,
              std::move(store_inputs), std::move(store_outputs),
              temp_buffer.view());

  if (log_xla.want_debug()) {
    for (auto&& input : inputs) {
      log_xla.debug() << compiler->Name() << " has input " << input.impl->name
                      << ", " << input.impl->array->shape()
                      << " with postcondition "
                      << input.impl->array.Precondition();
    }

    for (auto&& output : outputs) {
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

void ZukuExecuteContextImpl::RunAfterAllTasks(int64_t local_device_id,
                                              std::function<void()> on_done) {
  if (on_done) {
    after(last_execute_events_[local_device_id])
        .defer([](std::function<void()> callback) { callback(); },
               std::move(on_done));
  }
}

void ZukuExecuteContextImpl::Destroy(StoreHandle& store) {
  if (runtime_stopped.load() == false) {
    // no need to synchronize -- just removing the reference
    store.impl = nullptr;
  }  // the runtime no longer exists, we can't delete
}

void ZukuExecuteContextImpl::StoreBufferAction(int64_t local_device_id,
                                               BufferAction* action,
                                               const StoreHandle& store,
                                               bool blocking) {
  zuku::Processor p = LocalProcessor(local_device_id);
  auto token = on(p).defer(
      [](int64_t local_device_id, BufferAction* action,
         zuku::ShardedArray& array) {
        // ApplyStoreBufferAction(local_device_id, action, array);
      },
      local_device_id, action, store.impl->array);

  if (blocking) {
    token.Wait();
  }
}

bool ZukuExecuteContextImpl::HasLocalShard(const StoreHandle& handle) {
  return handle.impl->array->HasTile();
}

void* ZukuExecuteContextImpl::SliceLocalShard(int64_t local_device_id,
                                              const StoreHandle& handle) {
  zuku::Processor p = LocalProcessor(local_device_id);
  auto [buffer] = on(p).defer(
      [](const zuku::ShardedArray& array) {
        // do the slicing
        return array.tile().data();
      },
      handle.impl->array);
  return const_cast<void*>(buffer.wait_and_get());
}

zuku::ShardedShape ZukuExecuteContextImpl::GetStoreShardedShape(
    const StoreHandle& handle) {
  return handle.impl->array->shape();
}

StoreHandle ZukuExecuteContextImpl::AssembleShardsImpl(
    int64_t local_device_id, int64_t global_device_id, zuku::ShardedShape shape,
    Shard shard, std::shared_ptr<MultiMeshStream> stream,
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
              [](std::shared_ptr<MultiMeshStream> stream, zuku::Processor p,
                 Shard shard, zuku::ShardedArray& array) {
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

void ZukuExecuteContextImpl::Clear() {
  host_caches_.clear();
  device_caches_.clear();
}

void ZukuExecuteContextImpl::Rename(StoreHandle& handle, std::string name) {
  log_xla.debug() << "renaming " << handle.impl->name << " to " << name
                  << ", tensor_id=" << handle.impl->array->mesh_unique_id();
  handle.impl->name = std::move(name);
}

void ZukuExecuteContextImpl::Reshard(int64_t local_device_id,
                                     int64_t global_device_id,
                                     const StoreHandle& src,
                                     const StoreHandle& dst) {
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

bool ZukuExecuteContextImpl::IsGpu() {
  return zuku::Processor::DefaultType() == zuku::Processor::Type::GPU;
}

void ZukuExecuteContextImpl::StartTimer(const std::string& name) {
  auto [start_time] = after(last_execute_events_[0]).defer([] {
    return std::chrono::steady_clock::now();
  });
  pending_timers_.emplace(name, std::move(start_time));
}

void ZukuExecuteContextImpl::StopTimer(const std::string& name) {
  auto iter = pending_timers_.find(name);
  if (iter == pending_timers_.end()) {
    log_xla.warning() << "cannot stop timer " << name << ", does not exist";
    return;
  }

  after(last_execute_events_[0])
      .defer(
          [](std::string name, std_timer start_timer) {
            auto stop_timer = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = stop_timer - start_timer;
            log_xla.info() << name << " finished in " << diff.count() << "s";
          },
          std::move(iter->first), std::move(iter->second));

  pending_timers_.erase(iter);
}

zuku::Future<zuku::ArrayTile> ZukuExecuteContextImpl::CreateBuffer(
    int64_t local_device_id, int64_t global_device_id, int64_t size) {
  // Realm currently does not build with int64_t types, which creates
  // build issues on some systems
  // NOLINTNEXTLINE(google-runtime-int)
  Realm::Rect<1, long long> bounds;
  bounds.lo[0] = 0;
  bounds.hi[0] = std::max<int64_t>(0, size - 1);
  zuku::TileShape shape{.type = zuku::SupportedType::S8,
                        .dims = {size},
                        .bounds = std::move(bounds)};
  zuku::Processor p = LocalProcessor(local_device_id);
  return zuku::ArrayTile::CreateFuture(std::move(shape),
                                       {.processor = std::move(p)});
}

void ZukuExecuteContextImpl::Free(int64_t local_device_id, StoreHandle handle,
                                  bool keep_in_cache) {
  auto shape = handle.impl->array->shape();
  log_xla.debug() << "FreeStore: device=" << local_device_id
                  << ", shape=" << shape;
  if (keep_in_cache) {
    device_caches_[local_device_id].Free(shape, std::move(handle.impl->array));
  } else {
    // release the handle (explicitly done for clarity)
    handle.impl = nullptr;
  }
}

void ZukuExecuteContextImpl::ClearStoreCache(int64_t local_device_id) {
  device_caches_[local_device_id].Clear();
}

StoreHandle ZukuExecuteContextImpl::CreateStoreImpl(int64_t local_device_id,
                                                    int64_t global_device_id,
                                                    zuku::ShardedShape shape,
                                                    CreateStoreConfig config) {
  log_xla.debug() << "CreateStore: device=" << local_device_id
                  << ", shape=" << shape
                  << ", name=" << config.name.value_or("anonymous");

  auto array = [&] {
    if (config.allocate_from_cache) {
      return device_caches_[local_device_id].Get(shape, config.min_cache_size);
    }
    // use the cache as a store allocator
    return device_caches_[local_device_id].Make(shape);
  }();

  const int64_t next_id = NextStoreId();

  std::string array_name = [&] {
    if (config.name.has_value()) {
      return *std::move(config.name);
    }
    return std::string("anonymous");
  }();

  return StoreHandle{
      .impl = std::shared_ptr<StoreHandleImpl>(new StoreHandleImpl{
          .name = std::move(array_name), .array = std::move(array)}),
      .unique_id = next_id};
}

void ZukuExecuteContextImpl::FenceCompilation() {
  Realm::Event all_compilations =
      Realm::Event::merge_events(pending_compilation_events_);
  pending_compilation_events_.clear();
  all_compilations.wait();
}

std::shared_ptr<ZukuExecuteContext> ZukuExecuteContextImpl::Create(
    zuku::RealmConfig config) {
  std::array argv = {"multimesh-jax"};
  if (!rt.has_value()) {
    rt = zuku::Init(argv.size(), (char**)argv.data(), config);
    // This has to come after PyFinalize
    atexit(StopZuku);
  }

  if (keep_from_deleting == nullptr) {
    keep_from_deleting = new std::vector<zuku::Store<zuku::ShardedArray>>;
  }

  if (keep_tiles_from_deleting == nullptr) {
    keep_tiles_from_deleting = new std::vector<zuku::Future<zuku::ArrayTile>>;
  }

  return std::shared_ptr<ZukuExecuteContext>(new ZukuExecuteContextImpl{*rt});
}

ZukuExecuteContextImpl::~ZukuExecuteContextImpl() { all_contexts_.erase(this); }

void ZukuExecuteContextImpl::ClearAllContexts() {
  for (auto* context : all_contexts_) {
    context->Clear();
  }
}

ZukuExecuteContextImpl::ZukuExecuteContextImpl(Realm::Runtime rt) {
  const auto& procs = zuku::Processor::DefaultProcs();
  zuku::Processor host = zuku::Processor::Util();
  last_execute_events_.reserve(procs.size());
  last_control_events_.reserve(procs.size());
  device_caches_.reserve(procs.size());
  host_caches_.reserve(procs.size());
  for (auto&& proc : procs) {
    last_execute_events_.push_back(Realm::Event::NO_EVENT);
    last_control_events_.push_back(Realm::UserEvent::NO_USER_EVENT);
    zuku::Processor host =
        zuku::Processor::Create(proc.id(), zuku::Processor::Type::CPU);
    host_caches_.emplace_back(Realm::Memory::Kind::Z_COPY_MEM, host);
    device_caches_.emplace_back(proc.DefaultMemoryKind(), proc);
  }
  all_contexts_.insert(this);
}

std::set<ZukuExecuteContextImpl*> ZukuExecuteContextImpl::all_contexts_;

}  // namespace xla

extern "C" void ZukuShutdown() {
  // Make sure to clear all handles held
  // so that nothing gets deleted during program cleanup
  xla::ZukuExecuteContextImpl::ClearAllContexts();
  StopZuku();
  xla::ClearCachedStreams();
}
