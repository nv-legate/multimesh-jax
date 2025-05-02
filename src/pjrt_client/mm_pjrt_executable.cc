/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mm_pjrt_executable.h"

#include <functional>
#include <variant>

#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/pjrt/multimesh/mm_pjrt_buffer.h"
#include "xla/pjrt/multimesh/mm_sharding.h"
#include "xla/pjrt/multimesh/store_handle_fwd.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/buffer_assignment.h"
#include "xla/util.h"

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

namespace xla {

template <typename H>
H AbslHashValue(H h, const StoreHandle& store) {
  return H::combine(std::move(h), store.unique_id);
}

namespace {

constexpr absl::string_view kDynamicSchedulePostLoopTasks =
    "MULTIMESH_DYNAMIC_SCHEDULE_POST_LOOP_TASKS";

std::atomic<int64_t> next_run_id{0};

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
WrapperPjRtExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  TF_ASSIGN_OR_RETURN(
      auto outputs, wrapped_->ExecuteSharded(argument_handles, device, options,
                                             returned_future, fill_future));
  return Wrap(std::move(outputs));
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
WrapperPjRtExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  TF_ASSIGN_OR_RETURN(
      auto outputs, wrapped_->ExecutePortable(argument_handles, device, options,
                                              returned_future, fill_future));
  return Wrap(std::move(outputs));
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
WrapperPjRtExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  VLOG(3) << "WrapperPjRtExecutable::Execute: " << name();

  std::vector<std::vector<PjRtBuffer*>> unwrapped_arguments;
  unwrapped_arguments.reserve(argument_handles.size());
  for (const auto& handle_vec : argument_handles) {
    std::vector<PjRtBuffer*> unwrapped_handles;
    unwrapped_handles.reserve(handle_vec.size());
    for (auto* buf : handle_vec) {
      auto* mm_buffer = dynamic_cast<MultiMeshPjRtBuffer*>(buf);
      if (mm_buffer) {
        if (mm_buffer->has_host_action()) {
          TF_RETURN_IF_ERROR(mm_buffer->ResolveHostAction());
        }
        if (!mm_buffer->has_native_buffer()) {
          return InvalidArgumentStrCat(
              "WrapperPjRtExecutable::Execute: received multimesh buffer not "
              "backed by native buffer");
        }
        unwrapped_handles.push_back(mm_buffer->native_buffer());
      } else {
        unwrapped_handles.push_back(buf);
      }
    }
    unwrapped_arguments.push_back(std::move(unwrapped_handles));
  }

  TF_ASSIGN_OR_RETURN(auto outputs,
                      wrapped_->Execute({unwrapped_arguments.data(),
                                         unwrapped_arguments.size()},
                                        options, returned_futures));
  return Wrap(std::move(outputs));
}

std::vector<std::unique_ptr<PjRtBuffer>> WrapperPjRtExecutable::Wrap(
    std::vector<std::unique_ptr<PjRtBuffer>> outputs) {
  std::vector<std::unique_ptr<PjRtBuffer>> wrapped;
  wrapped.reserve(outputs.size());

  auto get_global_shape = [&](int index) -> Shape {
    if (program_shape_.result().IsTuple()) {
      return program_shape_.result().tuple_shapes(index);
    }
    return program_shape_.result();
  };

  int output_index = 0;
  for (auto& buf : outputs) {
    auto device_shape = buf->on_device_shape();
    auto global_shape = get_global_shape(output_index);
    auto* device = buf->device();

    PjRtMemorySpace* memory_space = buf->memory_space();
    auto mm_buffer = std::make_unique<MultiMeshPjRtBuffer>(
        std::move(buf), std::nullopt, std::move(global_shape),
        std::move(device_shape), mm_client_, wrapped_->client(), device,
        memory_space, "wrapped");
    wrapped.push_back(std::move(mm_buffer));
    ++output_index;
  }
  return std::move(wrapped);
}

std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>
WrapperPjRtExecutable::Wrap(
    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> outputs) {
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> wrapped;
  wrapped.reserve(outputs.size());
  for (auto& vec : outputs) {
    wrapped.push_back(Wrap(std::move(vec)));
  }
  return std::move(wrapped);
}

class TaskTempMemoryAllocator {
 public:
  TaskTempMemoryAllocator(absl::Span<const BufferAllocation> allocations,
                          TaskMemoryAllocator* allocator,
                          bool tupled_args = false, size_t num_args = 0)
      : allocator_(allocator), position_(0) {
    total_size_ = 0;
    if (tupled_args) {
      size_t alloc_size = PaddedSize(num_args * sizeof(void*));
      total_size_ += alloc_size;
    }
    for (const BufferAllocation& alloc : allocations) {
      if (alloc.IsPreallocatedTempBuffer()) {
        // Round to an increment of padding
        total_size_ += PaddedSize(alloc.size());
      }
    }
    buffer_ = (char*)allocator->Allocate(total_size_);
  }

  void* Next(size_t alloc_size) {
    void* buf = buffer_ + position_;
    position_ += PaddedSize(alloc_size);
    return buf;
  }

 private:
  static size_t PaddedSize(size_t size) { return (size + padding) & ~padding; }

  // The -1 enables bit twiddling
  static constexpr size_t padding = 4096 - 1;

  size_t position_;
  size_t total_size_;
  char* buffer_;
  // field is used, but is somehow not recognized by clang-tidy
  // NOLINTNEXTLINE(clang-diagnostic-unused-private-field)
  TaskMemoryAllocator* allocator_;
};

PjRtClient* MultiMeshPjRtExecutable::client() const { return mm_client_; }

int MultiMeshPjRtExecutable::num_replicas() const {
  return device_assignment_->replica_count();
}

int MultiMeshPjRtExecutable::num_partitions() const {
  return device_assignment_->computation_count();
}

int64_t MultiMeshPjRtExecutable::SizeOfGeneratedCodeInBytes() const {
  return 0;
}

const DeviceAssignment& MultiMeshPjRtExecutable::device_assignment() const {
  return *device_assignment_;
}

absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
MultiMeshPjRtExecutable::addressable_device_logical_ids() const {
  return addressable_device_logical_ids_;
}

absl::Span<PjRtDevice* const> MultiMeshPjRtExecutable::addressable_devices()
    const {
  return addressable_devices_;
}

MultiMeshPjRtExecutable::MultiMeshPjRtExecutable(
    MultiMeshClient* mm_client, PjRtClient* base_client, absl::string_view name,
    ProgramShape program_shape, std::vector<MpmdOperation> schedule,
    std::vector<Store> temporaries,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, const Shape& result_shape,
    const std::vector<Layout>& parameter_layouts,
    const std::vector<Layout>& output_layouts,
    std::optional<std::vector<OpSharding>> parameter_shardings,
    const std::vector<Shape>& output_shapes,
    std::optional<std::vector<OpSharding>> output_shardings,
    std::unique_ptr<WrapperPjRtExecutable> fast_path_exe)
    : mm_client_(mm_client),
      program_shape_(std::move(program_shape)),
      base_client_(base_client),
      schedule_(std::move(schedule)),
      temporaries_(std::move(temporaries)),
      device_assignment_(std::move(device_assignment)),
      name_(std::string(name)),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)),
      result_shape_(result_shape),
      parameter_layouts_(parameter_layouts),
      output_layouts_(output_layouts),
      parameter_shardings_(std::move(parameter_shardings)),
      output_shapes_(output_shapes),
      output_shardings_(std::move(output_shardings)),
      context_(mm_client->shared_context()),
      fast_path_exe_(std::move(fast_path_exe)) {}

void MultiMeshPjRtExecutable::Delete() { deleted_ = true; }

bool MultiMeshPjRtExecutable::IsDeleted() { return deleted_; }

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
MultiMeshPjRtExecutable::GetHloModules() const {
  return Unimplemented("MultiMeshPjRtExecutable::GetHloModules");
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
MultiMeshPjRtExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  LOG(ERROR) << "Unexpected call to MultiMeshPjRtExecutable::ExecuteSharded.";
  return Unimplemented("ExecuteSharded");
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
MultiMeshPjRtExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  LOG(ERROR) << "Unexpected call to MultiMeshPjRtExecutable::ExecutePortable.";
  return Unimplemented("ExecutePortable");
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
MultiMeshPjRtExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  VLOG(1) << "MultiMeshPjRtExecutable::Execute called for '" << name();

  // for testing purposes, we may invoke this function
  // wither fewer argument handles than there are addressable devices
  const int64_t num_local_devices = argument_handles.size();

  const int64_t start_device = (*device_assignment_)(0, 0);
  const int64_t num_devices = device_assignment_->num_elements();
  zuku::DeviceList devices_for_exe{
      {.start = start_device, .num_devices = num_devices}};

  struct ResultShape {
    Shape global_shape;
    Shape local_shape;
  };

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> results(
      num_local_devices);

  if (argument_handles.empty()) {
    return InvalidArgumentStrCat("no argument vectors passed to executable ",
                                 name());
  }

  const bool static_schedule_all_tasks = [&] {
    const char* env = getenv(kDynamicSchedulePostLoopTasks.data());
    if (env) {
      return std::atoi(env) == 0;
    }
    return true;
  }();

  size_t num_handles = argument_handles[0].size();

  bool all_native_buffer_inputs = true;
  for (auto&& argument_vec : argument_handles) {
    for (PjRtBuffer* buffer : argument_vec) {
      auto* mm_buffer = dynamic_cast<MultiMeshPjRtBuffer*>(buffer);
      if (!mm_buffer) {
        return InvalidArgumentStrCat(
            "buffer passed to MultiMesh executable that is not a MultiMesh "
            "buffer");
      }
      all_native_buffer_inputs =
          all_native_buffer_inputs && mm_buffer->has_native_buffer();
      if (fast_path_exe_ && !mm_buffer->has_native_buffer()) {
        VLOG(3) << mm_buffer->name()
                << " is not a native buffer in input to fast path exe "
                << fast_path_exe_->name() << std::endl;
      }
    }
  }

  if (fast_path_exe_ && all_native_buffer_inputs) {
    VLOG(3) << "Executing fast path " << fast_path_exe_->name();
    // we don't have any stores to process and the executable is small
    // and already compiled we can take the "fast" path exe and just push
    // everything through native backends
    const auto& task = std::get<SpmdHloModuleTask>(schedule_[0].op);
    std::vector<std::vector<PjRtBuffer*>> buffers;
    buffers.reserve(num_local_devices);
    for (auto&& handles : argument_handles) {
      buffers.emplace_back();
      buffers.back().resize(handles.size());
      int index = 0;
      for (auto&& input : task.inputs) {
        // inputs may get permuted from the compilation process
        buffers.back()[input.index] = handles[index++];
      }
    }

    TF_ASSIGN_OR_RETURN(auto outputs, fast_path_exe_->Execute(
                                          buffers, options, returned_futures));

    std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> permuted_outputs;
    permuted_outputs.reserve(outputs.size());
    for (auto&& output_vec : outputs) {
      permuted_outputs.emplace_back();
      permuted_outputs.back().resize(output_vec.size());
      int index = 0;
      for (auto&& output : task.outputs) {
        permuted_outputs.back()[output.index] = std::move(output_vec[index++]);
      }
    }

    return std::move(permuted_outputs);
  }

  context_->OpenWindow();
  context_->StartTimer(name_);

  // prepare callbacks
  auto compute_callbacks =
      std::make_unique<std::vector<std::function<void()>>>();
  if (returned_futures.has_value()) {
    for (size_t idx = 0; idx < num_local_devices; ++idx) {
      auto promise = PjRtFuture<>::CreatePromise();
      auto future = PjRtFuture<>(promise);
      compute_callbacks->push_back([promise = std::move(promise)]() mutable {
        promise.Set(absl::OkStatus());
      });
      returned_futures->push_back(std::move(future));
    }
  }

  std::vector<std::vector<StoreHandle>> parameter_stores(num_local_devices);

  auto get_sharding_devices = [&](const HloSharding& sharding) {
    if (sharding.IsReplicated()) {
      return devices_for_exe;
    }
    return zuku::DeviceList{
        {.start = sharding.tile_assignment().first(),
         .num_devices = sharding.tile_assignment().num_elements()}};
  };

  // prepare the parameter stores
  auto mm_stream = std::make_shared<MultiMeshStream>(mm_client_->backend());
  OpSharding dummy_sharding;
  for (size_t local_device = 0; local_device < num_local_devices;
       ++local_device) {
    auto& arguments = argument_handles[local_device];
    parameter_stores[local_device].reserve(arguments.size());
    int64_t index = 0;
    for (PjRtBuffer* const input : arguments) {
      auto* mm_buffer = dynamic_cast<MultiMeshPjRtBuffer*>(input);

      if (mm_buffer->has_store()) {
        zuku::ShardedShape sharded_shape =
            context_->GetStoreShardedShape(mm_buffer->store());
        VLOG(3) << "initialized Arg_" << index
                << " with shape=" << sharded_shape;
        parameter_stores[local_device].push_back(mm_buffer->store());
      } else {
        const OpSharding& op_sharding = [&] {
          if (parameter_shardings_.has_value()) {
            return (*parameter_shardings_)[index];
          }
          return dummy_sharding;
        }();

        TF_ASSIGN_OR_RETURN(HloSharding global_sharding,
                            HloSharding::FromProto(op_sharding));

        TF_ASSIGN_OR_RETURN(
            zuku::ShardedShape sharded_shape,
            XlaShapeToZukuShape(mm_buffer->on_device_shape(),
                                get_sharding_devices(global_sharding),
                                global_sharding,
                                /*global_shape*/ false));
        TF_ASSIGN_OR_RETURN(auto store,
                            mm_buffer->ToStore(sharded_shape, mm_stream));
        parameter_stores[local_device].push_back(std::move(store));
      }
      ++index;
    }
  }

  // prepare the root stores
  size_t num_results =
      result_shape_.IsTuple() ? result_shape_.tuple_shapes_size() : 1;
  std::vector<ResultShape> out_shapes(num_results);
  std::vector<std::optional<std::string>> output_names(num_results);
  std::vector<std::vector<StoreHandle>> root_stores(num_local_devices);
  int64_t local_index = 0;
  for (int64_t local_device = 0; local_device < num_local_devices;
       ++local_device) {
    root_stores[local_device].resize(num_results);
  }

  int64_t max_buffer_temp_required = 0;
  int64_t max_num_scalars = 0;
  int64_t max_num_outputs = 0;
  auto add_max_temp_buffer = [&](const SpmdHloModuleTask& task) {
    int64_t task_scalars = 0;
    for (auto&& input : task.inputs) {
      if (input.scalar) {
        task_scalars++;
      }
    }
    max_num_scalars = std::max(max_num_scalars, task_scalars);
    if (task.compiler->CompiledLocally()) {
      max_buffer_temp_required =
          std::max(max_buffer_temp_required, task.compiler->TempRequired());
    }
    max_num_outputs = std::max<int64_t>(max_num_outputs, task.outputs.size());
  };

  auto get_shape_at_index = [&](const Shape& shape, size_t idx) {
    if (shape.IsTuple()) {
      return shape.tuple_shapes(idx);
    }
    return shape;
  };

  auto add_root_store = [&](const Store& output, const ResultShape& shape,
                            std::optional<int64_t> parameter_number_alias) {
    // already configured
    if (root_stores[0][output.index].impl) {
      return;
    }
    out_shapes[output.index] = shape;
    output_names[output.index] = output.name;

    VLOG(3) << "looking for alias of global=" << output.index
            << ",local=" << local_index << " " << output.shape;
    if (parameter_number_alias.has_value()) {
      VLOG(3) << "root global=" << output.index << ",local=" << local_index
              << " " << output.name << " " << output.shape
              << " aliases parameter " << *parameter_number_alias;
      for (int64_t local_device = 0; local_device < num_local_devices;
           ++local_device) {
        root_stores[local_device][output.index] =
            parameter_stores[local_device][*parameter_number_alias];
      }
    } else {
      for (int64_t local_device = 0; local_device < num_local_devices;
           ++local_device) {
        VLOG(3) << "root global=" << output.index << ",local=" << local_index
                << " " << output.shape << " is new output";
        root_stores[local_device][output.index] = context_->CreateStore(
            local_device,
            addressable_devices_[local_device]->global_device_id().value(),
            output.sharded_shape, {.name = output.name});
      }
    }
  };

  auto setup_root_stores = [&](const SpmdHloModuleTask& task) {
    Shape spmd_root_shape = GetSpmdShape(
        task.module->module->entry_computation()->root_instruction());

    auto get_root_shape = [&](size_t global_root,
                              size_t task_root) -> ResultShape {
      return {.global_shape = get_shape_at_index(result_shape_, global_root),
              .local_shape = get_shape_at_index(spmd_root_shape, task_root)};
    };

    int64_t local_index = 0;
    for (const Store& output : task.outputs) {
      if (output.type == Store::Type::ROOT) {
        auto parameter_number_alias = [&]() -> std::optional<int64_t> {
          auto local_alias = task.compiler->OutputAlias(local_index);
          if (local_alias.has_value() &&
              task.inputs[*local_alias].type == Store::Type::PARAM) {
            return task.inputs[*local_alias].index;
          }
          return std::nullopt;
        }();
        add_root_store(output, get_root_shape(output.index, local_index),
                       parameter_number_alias);
      }
      ++local_index;
    }
  };

  for (const MpmdOperation& op : schedule_) {
    std::visit(
        overloaded{
            [&](const SpmdHloModuleTask& task) { setup_root_stores(task); },
            [&](const Reshard& reshard) {
              if (reshard.output.type == Store::Type::ROOT) {
                auto spmd_shape = spmd::MakePartitionedShape(
                    reshard.output.shape, reshard.output.mpmd_sharding);
                add_root_store(reshard.output,
                               {reshard.output.shape, spmd_shape},
                               std::nullopt);
              }
            },
            [&](const auto&) {}},
        op.op);
  }

  // prepare the temporary stores
  if (cached_temporary_stores_.empty()) {
    cached_temporary_stores_.resize(num_local_devices);
    for (int64_t local_device = 0; local_device < num_local_devices;
         ++local_device) {
      cached_temporary_stores_[local_device].reserve(temporaries_.size());
      for (auto&& store : temporaries_) {
        cached_temporary_stores_[local_device].push_back(context_->CreateStore(
            local_device,
            addressable_devices_[local_device]->global_device_id().value(),
            store.sharded_shape, {.name = store.name}));
      }
    }
  }

  // compute the amount of temp buffer space required
  for (const MpmdOperation& op : schedule_) {
    std::visit(overloaded{[&](const SpmdHloModuleTask& task) {
                            add_max_temp_buffer(task);
                          },
                          [&](const auto&) {}},
               op.op);
  }

  if (temp_allocations_.empty()) {
    int64_t max_scalar_temp_required = max_num_scalars * kTempMinAlignment;
    int64_t max_tuple_temp_required =
        AlignTempSize(max_num_outputs * sizeof(void*));
    max_buffer_temp_required = AlignTempSize(max_buffer_temp_required);
    const int64_t total_temp_required = max_scalar_temp_required +
                                        max_buffer_temp_required +
                                        max_tuple_temp_required;

    temp_allocations_.reserve(num_local_devices);

    for (int64_t local_device = 0; local_device < num_local_devices;
         ++local_device) {
      temp_allocations_.push_back(context_->CreateBuffer(
          addressable_devices_[local_device]->local_device_id().value(),
          addressable_devices_[local_device]->global_device_id().value(),
          total_temp_required));
    }
  }

  auto get_store_handle = [&](int64_t local_device_id, const Store& store) {
    VLOG(5) << "Have store type=" << store.type << " index=" << store.index;
    switch (store.type) {
      case Store::Type::PARAM:
        return parameter_stores[local_device_id][store.index];
      case Store::Type::ROOT:
        return root_stores[local_device_id][store.index];
      case Store::Type::TEMP:
        return cached_temporary_stores_[local_device_id][store.index];
    }
  };

  auto get_store_handles = [&](int64_t local_device_id,
                               const std::vector<Store>& stores) {
    std::vector<StoreHandle> handles;
    handles.reserve(stores.size());
    for (auto&& store : stores) {
      if (!store.scalar) {
        handles.push_back(get_store_handle(local_device_id, store));
      }
    }
    return handles;
  };

  for (MpmdOperation& op : schedule_) {
    const int64_t run_id = next_run_id.fetch_add(int64_t(1));
    for (int64_t local_device = 0; local_device < num_local_devices;
         ++local_device) {
      std::visit(
          overloaded{
              [&](const SpmdHloModuleTask& task) {
                auto inputs = get_store_handles(local_device, task.inputs);
                auto outputs = get_store_handles(local_device, task.outputs);

                constexpr int kLoopPriority = 10;
                constexpr int kZeroPriority = 0;

                context_->CreateExecuteTask(
                    run_id,
                    addressable_devices_[local_device]
                        ->local_device_id()
                        .value(),
                    addressable_devices_[local_device]
                        ->global_device_id()
                        .value(),
                    task.device_assignment, task.compiler, task.scalars, inputs,
                    outputs, temp_allocations_[local_device],
                    {.strict_ordering = task.loop ||
                                        static_schedule_all_tasks ||
                                        task.compiler->Concurrent(),
                     .priority = (task.loop ? kLoopPriority : kZeroPriority)});
              },
              [&](const Reshard& reshard) {
                auto source = get_store_handle(local_device, reshard.input);
                auto target = get_store_handle(local_device, reshard.output);
                context_->Reshard(addressable_devices_[local_device]
                                      ->local_device_id()
                                      .value(),
                                  addressable_devices_[local_device]
                                      ->global_device_id()
                                      .value(),
                                  source, target);
              },
              [&](const auto&) {}},
          op.op);
    }
  }

  if (!compute_callbacks->empty()) {
    for (int64_t local_device = 0; local_device < num_local_devices;
         ++local_device) {
      context_->RunAfterAllTasks(
          addressable_devices_[local_device]->local_device_id().value(),
          std::move(compute_callbacks->at(local_device)));
    }
  }

  for (size_t out = 0; out < out_shapes.size(); ++out) {
    const auto& result_shape = out_shapes[out];
    OpSharding op_sharding = [&] {
      if (output_shardings_.has_value()) {
        return (*output_shardings_)[out];
      }
      return OpSharding{};
    }();
    TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                        HloSharding::FromProto(op_sharding));
    for (size_t comp = 0; comp < num_local_devices; ++comp) {
      VLOG(3) << "creating output " << output_names[out].value_or("anon")
              << " with global shape " << result_shape.global_shape
              << " and local shape " << result_shape.local_shape;

      auto mm_buffer = std::make_unique<MultiMeshPjRtBuffer>(
          std::move(root_stores[comp][out]), hlo_sharding,
          result_shape.global_shape, result_shape.local_shape, mm_client_,
          base_client_, addressable_devices_[comp],
          addressable_devices_[comp]->default_memory_space().value_or(nullptr),
          output_names[out]);
      if (!mm_buffer->on_device_shape().has_layout()) {
        return InternalStrCat(
            "output ", out, ", ", output_names[out].value_or("anonymous"),
            " for executable ", name_,
            " has no layout: shape=", out_shapes[out].global_shape.ToString());
      }
      if (mm_buffer->store().impl == nullptr) {
        return InternalStrCat("output ", out, " for executable ", name_,
                              " got a null Zuku store");
      }
      results[comp].push_back(std::move(mm_buffer));
    }
  }

  context_->StopTimer(name_);
  context_->CloseWindow();
  return results;
}

absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
MultiMeshPjRtExecutable::GetParameterLayouts() const {
  std::vector<std::shared_ptr<const PjRtLayout>> layouts;
  layouts.reserve(parameter_layouts_.size());
  for (auto&& layout : parameter_layouts_) {
    layouts.push_back(std::make_shared<PjRtLayout>(layout));
  }
  return std::move(layouts);
}

absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
MultiMeshPjRtExecutable::GetOutputLayouts() const {
  std::vector<std::shared_ptr<const PjRtLayout>> layouts;
  layouts.reserve(output_layouts_.size());
  for (auto&& layout : output_layouts_) {
    layouts.push_back(std::make_shared<PjRtLayout>(layout));
  }
  return std::move(layouts);
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
MultiMeshPjRtExecutable::GetOutputMemoryKinds() const {
  return Unimplemented("GetOutputMemoryKinds is not supported.");
}

}  // namespace xla
