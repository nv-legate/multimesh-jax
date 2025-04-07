/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "xla/pjrt/legate/legate_pjrt_client.h"

#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "tsl/platform/logging.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/legate_computation.h"
#include "xla/pjrt/legate/legate_pjrt_buffer.h"
#include "xla/pjrt/legate/legate_pjrt_executable.h"
#include "xla/pjrt/legate/legate_sharding.h"
#include "xla/pjrt/legate/legate_utils.h"
#include "xla/pjrt/legate/mpmd_partition.h"
#include "xla/pjrt/legate/zuku_execute_context.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/utils.h"
#include "xla/service/backend.h"
#include "xla/service/computation_placer.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/platform_util.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/util.h"

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
namespace xla {

namespace {

constexpr int kFastPathInstructionCutoff = 20;
bool enable_fast_path_exe{true};
std::optional<int64_t> replicated_parameter_num_elements_cutoff = 0;
std::optional<int64_t> recompute_from_arguments_if_cost_less_than{1024 * 1024};
bool only_fuse_loop_tasks{false};
bool use_task_fusion{true};

struct CompileOutput {
  std::vector<MpmdOperation> schedule;
  std::vector<Store> temporaries;
  std::vector<OpSharding> parameter_shardings;
  std::vector<OpSharding> root_shardings;
  std::vector<Shape> output_shapes;
  std::vector<Layout> parameter_layouts;
  std::vector<Layout> output_layouts;
  Shape output_shape;
  Shape executable_layout;
};

bool SmallModuleTask(const MpmdOperation& op) {
  return std::visit(
      overloaded{[](const SpmdHloModuleTask& task) {
                   return task.module->module->instruction_count() <
                          kFastPathInstructionCutoff;
                 },
                 [](const auto&) { return false; }},
      op.op);
}

bool IsSubmeshReplicatedScalar(const Store& store,
                               const zuku::DeviceList& devices) {
  if (store.mpmd_sharding.IsReplicated() || !store.shape.dimensions().empty()) {
    return false;
  }

  return store.mpmd_sharding.HasPartialReplication() &&
         store.mpmd_sharding.tile_assignment().dimensions().back() ==
             devices.size();
}

absl::StatusOr<CompileOutput> CreateTasks(
    const XlaComputation& computation, CompileOptions options,
    const std::shared_ptr<DeviceAssignment>& device_assignment,
    Backend* backend, PjRtClient* client, MpmdPartitionConfig config) {
  // adjust the layouts for devices (as done in PjRtStreamExecutorClient)
  std::vector<const Shape*> argument_layout_pointers;
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [transfer_manager = backend->transfer_manager()](Shape shape) {
        return transfer_manager->ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));

  size_t num_roots = 0;
  size_t num_parameters = 0;
  const HloComputationProto* entry_comp = nullptr;
  const HloInstructionProto* root_instr = nullptr;
  for (const auto& comp : computation.proto().computations()) {
    if (comp.id() == computation.proto().entry_computation_id()) {
      entry_comp = &comp;
      for (const auto& instr : comp.instructions()) {
        if (instr.opcode() == "parameter") {
          if (instr.shape().element_type() == PrimitiveType::TUPLE) {
            num_parameters = instr.shape().tuple_shapes_size();
          } else {
            ++num_parameters;
          }
        }
        if (instr.id() == comp.root_id()) {
          root_instr = &instr;
          if (instr.shape().element_type() == PrimitiveType::TUPLE) {
            num_roots = instr.shape().tuple_shapes_size();
          } else {
            num_roots = 1;
          }
        }
      }
    }
  }

  Shape executable_layout(*options.executable_build_options.result_layout());

  config.use_task_fusion = use_task_fusion;
  config.replicated_parameter_num_elements_cutoff =
      replicated_parameter_num_elements_cutoff;
  config.only_fuse_loop_tasks = only_fuse_loop_tasks;
  config.recompute_from_arguments_if_cost_less_than =
      recompute_from_arguments_if_cost_less_than;
  TF_ASSIGN_OR_RETURN(
      auto partition_outputs,
      MpmdPartition(computation.proto(), options, argument_layout_pointers,
                    executable_layout, config));

  auto [ops, modules, temporaries, unused_params] =
      std::move(partition_outputs);

  std::vector<OpSharding> root_tuple_shardings(num_roots);
  std::vector<Shape> output_shapes(num_roots);
  std::vector<OpSharding> parameter_shardings(num_parameters);
  std::vector<Layout> output_layouts(num_roots);

  // all parameters may not get used so we take parameter layouts
  // initially from the input arguments
  std::vector<Layout> parameter_layouts;
  parameter_layouts.reserve(num_parameters);
  for (auto* shape_ptr : argument_layout_pointers) {
    if (shape_ptr->has_layout()) {
      parameter_layouts.push_back(shape_ptr->layout());
    } else {
      parameter_layouts.emplace_back();
    }
  }

  HloSharding default_replicated = HloSharding::Replicate();

  auto add_root = [&](const Store& output, const Shape& spmd_shape) {
    VLOG(3) << "Have output root " << output.index << ", shape=" << output.shape
            << " for " << output.name
            << " with sharding=" << output.mpmd_sharding;
    root_tuple_shardings[output.index] = output.mpmd_sharding.ToProto();
    output_layouts[output.index] = spmd_shape.layout();
    output_shapes[output.index] = spmd_shape;
    VLOG(3) << "output " << output.index << " " << output.name << " has shape "
            << spmd_shape << " and layout " << spmd_shape.layout();
  };

  auto collect_shapes_and_shardings =
      [&](const SpmdHloModuleTask& task) -> absl::Status {
    const HloModule& module = *task.module->module;
    VLOG(3) << "LegateClient::Compile: partition "
            << task.module->module->name();
    // once more update layout -- as done in Service::BuildExecutables
    auto device_shape_representation_fn = [&](const Shape& shape) {
      return backend->compiler()->DefaultDeviceShapeRepresentation(shape);
    };
    xla::UpdateEntryComputationLayout(task.module->module.get(),
                                      device_shape_representation_fn);

    auto* root = module.entry_computation()->root_instruction();

    for (auto* param : module.entry_computation()->parameter_instructions()) {
      const auto& input = task.inputs[param->parameter_number()];
      if (input.type == Store::Type::PARAM) {
        parameter_shardings[input.index] = input.mpmd_sharding.ToProto();
        if (IsSubmeshReplicatedScalar(input, task.device_assignment)) {
          VLOG(3) << "Forcing global replication of scalar input " << input.name
                  << ", number=" << input.index;
          parameter_shardings[input.index] = OpSharding{};
        }
        VLOG(3) << "Assigning layout/sharding to parameter " << input.name
                << ", number=" << input.index << ": " << param->shape() << " "
                << input.mpmd_sharding.ToString();
        parameter_layouts[input.index] = param->shape().layout();
      } else {
        VLOG(3) << "Using sharding for temporary input " << input.name << ": "
                << input.mpmd_sharding.ToString();
      }
    }

    Shape spmd_root_shape = GetSpmdShape(
        task.module->module->entry_computation()->root_instruction());

    for (size_t idx = 0; idx < task.outputs.size(); ++idx) {
      const auto& output = task.outputs[idx];

      if (output.type == Store::Type::ROOT) {
        if (spmd_root_shape.IsTuple()) {
          add_root(output, spmd_root_shape.tuple_shapes(idx));
        } else {
          add_root(output, spmd_root_shape);
        }
      } else {
        VLOG(3) << "Using sharding for temporary output " << output.name << ": "
                << output.mpmd_sharding.ToString();
      }
    }
    return absl::OkStatus();
  };

  for (const Store& param : unused_params) {
    parameter_shardings[param.index] = param.mpmd_sharding.ToProto();
    parameter_layouts[param.index] = param.shape.layout();
  }

  for (auto& module : modules) {
    DumpHloModuleIfEnabled(*module->module, "before_optimizations");
  }

  absl::flat_hash_map<HloModule*, std::shared_ptr<LegateCompiler>>
      module_to_compiler;
  for (auto& module : modules) {
    Shape spmd_root_shape =
        GetSpmdShape(module->module->entry_computation()->root_instruction());
    TF_ASSIGN_OR_RETURN(auto compiler, LegateCompiler::Create(
                                           module->module->Clone(""),
                                           spmd_root_shape, backend, client));
    module_to_compiler[module->module.get()] = std::move(compiler);
  }

  for (auto& op : ops) {
    std::visit(
        overloaded{[&](SpmdHloModuleTask& task) {
                     TF_CHECK_OK(collect_shapes_and_shardings(task));
                     task.compiler =
                         module_to_compiler[task.module->module.get()];
                   },
                   [&](Reshard& reshard) {
                     if (reshard.input.type == Store::Type::PARAM) {
                       parameter_shardings[reshard.input.index] =
                           reshard.input.mpmd_sharding.ToProto();
                       parameter_layouts[reshard.input.index] =
                           reshard.input.shape.layout();
                       VLOG(3) << "Assigning layout/sharding to parameter "
                               << reshard.input.name
                               << ", number=" << reshard.input.index << ": "
                               << reshard.input.shape << " "
                               << reshard.input.mpmd_sharding.ToString();
                     }
                     if (reshard.output.type == Store::Type::ROOT) {
                       auto shape = spmd::MakePartitionedShape(
                           reshard.output.shape, reshard.output.mpmd_sharding);
                       add_root(reshard.output, shape);
                     }
                   },
                   [&](const auto&) {
                     // no-op for now
                   }},
        op.op);
  }

  Shape output_shape = [&] {
    if (root_instr->shape().element_type() == PrimitiveType::TUPLE) {
      xla::ShapeProto tuple_shape;
      tuple_shape.set_element_type(PrimitiveType::TUPLE);
      for (const auto& shape : output_shapes) {
        *tuple_shape.add_tuple_shapes() = shape.ToProto();
      }
      return Shape(tuple_shape);
    }
    return output_shapes[0];
  }();

  VLOG(3) << computation.proto().name() << " has output shape "
          << output_shape.ToString();

  return CompileOutput{.schedule = std::move(ops),
                       .temporaries = std::move(temporaries),
                       .parameter_shardings = std::move(parameter_shardings),
                       .root_shardings = std::move(root_tuple_shardings),
                       .output_shapes = std::move(output_shapes),
                       .parameter_layouts = std::move(parameter_layouts),
                       .output_layouts = std::move(output_layouts),
                       .output_shape = std::move(output_shape),
                       .executable_layout = std::move(executable_layout)};
}

}  // namespace

absl::StatusOr<PjRtDevice*> LegateClient::LookupDevice(
    PjRtGlobalDeviceId device_id) const {
  return base_client_->LookupDevice(device_id);
}

absl::StatusOr<PjRtDevice*> LegateClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_hardware_id) const {
  return base_client_->LookupAddressableDevice(local_hardware_id);
}

absl::string_view LegateClient::platform_version() const {
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
  return "legate " STRINGIFY(CUDART_VERSION);
  // return "0.0";
}

absl::StatusOr<DeviceAssignment> LegateClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return base_client_->GetDefaultDeviceAssignment(num_replicas, num_partitions);
}

absl::StatusOr<Layout> LegateClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  return base_client_->GetDefaultLayout(element_type, dims);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
LegateClient::CreateUninitializedBuffer(const Shape& shape,
                                        PjRtMemorySpace* memory_space) {
  return Unimplemented("CreateUninitializedBuffer not implemented on %s",
                       platform_name());
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
LegateClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const ShapeSpec> shapes,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  return Unimplemented(
      "CreateBuffersForAsyncHostToDevice not implemented on %s",
      platform_name());
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
LegateClient::CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                                PjRtMemorySpace* memory_space) {
  return Unimplemented(
      "CreateBuffersForAsyncHostToDevice not implemented on %s",
      platform_name());
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> LegateClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    LegateClient::HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory, const Layout* device_layout) {
  VLOG(3) << "LegateClient::BufferFromHostBuffer for memory "
          << memory->DebugString() << " with type " << (int)type << " and "
          << dims.size() << " dims, semantics=" << (int)host_buffer_semantics
          << ", data=" << data;

  Shape device_shape = ShapeUtil::MakeShape(type, dims);
  TF_ASSIGN_OR_RETURN(
      auto native_buf,
      base_client_->BufferFromHostBuffer(
          data, type, dims, byte_strides, host_buffer_semantics,
          std::move(on_done_with_host_buffer), memory, device_layout));
  auto buf = std::unique_ptr<PjRtBuffer>(std::make_unique<LegatePjRtBuffer>(
      std::move(native_buf), std::nullopt, device_shape, device_shape, this,
      base_client_.get(), memory->devices()[0], memory, "host-action"));

  return std::move(buf);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> LegateClient::BufferFromHostLiteral(
    const LiteralSlice& literal, PjRtMemorySpace* memory_space) {
  return Unimplemented("BufferFromHostLiteral not implemented on %s",
                       platform_name());
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
LegateClient::CreateViewOfDeviceBuffer(void* device_ptr, const Shape& shape,
                                       PjRtMemorySpace* memory_space,
                                       std::function<void()> on_delete_callback,
                                       std::optional<std::intptr_t> stream) {
  return Unimplemented("CreateViewOfDeviceBuffer not implemented on %s",
                       platform_name());
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
LegateClient::MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                                          PjRtDevice* device,
                                          PjRtCrossHostRecvNotifier notifier) {
  return Unimplemented("MakeCrossHostReceiveBuffers not implemented on %s",
                       platform_name());
}

absl::Status LegateClient::Defragment() { return absl::OkStatus(); }

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
LegateClient::DeserializeExecutable(absl::string_view serialized,
                                    std::optional<CompileOptions> options) {
  return Unimplemented("DeserializeExecutable not implemented on %s",
                       platform_name());
}

HloModuleProto LegateClient::ShardBatch(const HloModuleProto& proto) const {
  HloModuleProto sharded = proto;
  int num_devices = base_client_->device_count();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  for (HloComputationProto& comp : *sharded.mutable_computations()) {
    if (comp.id() == sharded.entry_computation_id()) {
      for (HloInstructionProto& instr : *comp.mutable_instructions()) {
        if (instr.opcode() == "parameter" &&
            instr.shape().dimensions_size() > 0 &&
            instr.shape().dimensions().at(0) >= num_devices) {
          OpSharding* new_sharding = instr.mutable_sharding();
          new_sharding->set_type(OpSharding::OTHER);
          new_sharding->mutable_tile_assignment_dimensions()->Clear();
          new_sharding->mutable_tile_assignment_dimensions()->Add(num_devices);
          for (int i = 1; i < instr.shape().dimensions_size(); i++) {
            new_sharding->mutable_tile_assignment_dimensions()->Add(1);
          }
          new_sharding->mutable_tile_assignment_devices()->Assign(
              devices.begin(), devices.end());
        }
      }
    }
  }
  return sharded;
}

static absl::StatusOr<HloInstructionProto*> GetMutableRoot(
    HloModuleProto& proto) {
  for (auto& comp : *proto.mutable_computations()) {
    if (comp.id() == proto.entry_computation_id()) {
      for (auto& instr : *comp.mutable_instructions()) {
        if (instr.id() == comp.root_id()) {
          return &instr;
        }
      }
    }
  }
  return InvalidArgument(
      "HloModuleProto has not root instruction of entry computation");
}

absl::Status AssignExplicitParameterSharding(
    bool auto_shard, const HloModuleProto& proto,
    std::vector<OpSharding>& param_shardings) {
  // after assigning the shardings from the individual tasks, make sure that the
  // param shardings agree with any explicitly given input shardings
  for (const auto& comp : proto.computations()) {
    if (comp.id() == proto.entry_computation_id()) {
      for (const auto& instr : comp.instructions()) {
        if (instr.opcode() == "parameter") {
          if (instr.shape().element_type() == PrimitiveType::TUPLE) {
            if (instr.has_sharding()) {
              size_t idx = 0;
              for (const auto& op_sharding :
                   instr.sharding().tuple_shardings()) {
                if (!auto_shard ||
                    op_sharding.type() != OpSharding::REPLICATED) {
                  param_shardings[idx] = op_sharding;
                }
                ++idx;
              }
            }
          } else {
            if (!auto_shard || instr.has_sharding()) {
              param_shardings[instr.parameter_number()] = instr.sharding();
            }
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LegateClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  return Compile(computation, options, std::nullopt,
                 /*only_compile_device0=*/false);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LegateClient::Compile(
    const XlaComputation& computation, CompileOptions options,
    std::optional<int64_t> max_per_process, bool only_compile_device0,
    MpmdPartitionConfig config) {
  TF_ASSIGN_OR_RETURN(auto program_shape, computation.GetProgramShape());
  VLOG(3) << "LegateClient::Compile module for '" << computation.name()
          << "' with initial result shape = " << program_shape.result();

  int num_replicas;
  int num_partitions;
  std::shared_ptr<DeviceAssignment> device_assignment;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      options.compile_portable_executable, &options.executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));

  if (device_assignment) {
    VLOG(3) << computation.proto().name() << " has device assignment "
            << device_assignment->ToString()
            << " passed into LegateClient::Compile" << std::endl;

    absl::flat_hash_set<int64_t> local_ids;
    int64_t num_processes = device_count() / addressable_device_count();
    std::vector<int64_t> num_per_process(num_processes, 0);
    if (!max_per_process.has_value()) {
      max_per_process = 0;
      for (auto&& id : *device_assignment) {
        auto* dev = devices()[id];
        num_per_process[dev->process_index()]++;
        max_per_process =
            std::max(num_per_process[dev->process_index()], *max_per_process);
      }
    }

    if ((device_assignment->num_elements() == 1 && device_count() > 1) ||
        (max_per_process == 1 && addressable_device_count() > 1)) {
      // running on a single device subset requires us to dispatch directly to
      // the GPU stack to avoid control replication overheads
      TF_ASSIGN_OR_RETURN(auto program_shape, computation.GetProgramShape());
      TF_ASSIGN_OR_RETURN(auto exe,
                          base_client_->Compile(computation, options));
      return std::make_unique<WrapperPjRtExecutable>(std::move(exe), this,
                                                     std::move(program_shape));
    }

    if (*max_per_process < addressable_device_count() &&
        *max_per_process < device_assignment->num_elements()) {
      return InvalidArgumentStrCat(
          "LegatePjRtClient::Compile: cannot create executable on a subset of "
          "GPUs per node across multiple nodes");
    }
  }

  TF_ASSIGN_OR_RETURN(auto compile_result,
                      CreateTasks(computation, options, device_assignment,
                                  backend(), base_client_.get(), config));

  std::vector<PjRtStreamExecutorLoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_ids;
  std::vector<PjRtDevice*> addressable_devices;
  if (device_assignment != nullptr) {
    addressable_device_logical_ids.reserve(num_replicas * num_partitions);
    addressable_devices.reserve(num_replicas * num_partitions);
    if (only_compile_device0) {
      PjRtLoadedExecutable::LogicalDeviceIds logical_device_id;
      TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                          LookupDevice(PjRtGlobalDeviceId(0)));
      logical_device_id.replica = 0;
      logical_device_id.partition = 0;
      addressable_device_logical_ids.push_back(std::move(logical_device_id));
      addressable_devices.push_back(device);
    } else {
      for (int replica = 0; replica < num_replicas; ++replica) {
        for (int partition = 0; partition < num_partitions; ++partition) {
          int device_id = (*device_assignment)(replica, partition);
          TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                              LookupDevice(PjRtGlobalDeviceId(device_id)));
          if (device->process_index() == this->process_index()) {
            PjRtLoadedExecutable::LogicalDeviceIds logical_device_id;
            logical_device_id.replica = replica;
            logical_device_id.partition = partition;
            addressable_device_logical_ids.push_back(
                std::move(logical_device_id));
            addressable_devices.push_back(device);
          }
        }
      }
    }

    if (addressable_devices.empty()) {
      return InvalidArgument(
          "Device assignment (%s) does not have any local devices.",
          device_assignment->ToString());
    }
  }

  std::unique_ptr<WrapperPjRtExecutable> fast_path_exe{nullptr};
  if (enable_fast_path_exe && compile_result.schedule.size() == 1 &&
      SmallModuleTask(compile_result.schedule.front()) &&
      !ContainsLegateCustomCall(computation.proto())) {
    VLOG(3) << "LegateClient::Compile: fast path taken for "
            << computation.name();

    const auto& task =
        std::get<SpmdHloModuleTask>(compile_result.schedule.front().op);

    // We have derived a sharding pattern that must be respected
    // so we use the new hlo module for the task in the computation
    XlaComputation sharded_computation{task.compiler->module().ToProto()};
    CompileOptions sharded_options{options};
    // all shardings should have been filled in by now, if any shardings get
    // overwritten from here it can create problems later
    sharded_options.executable_build_options
        .set_allow_spmd_sharding_propagation_to_parameters({false});

    TF_ASSIGN_OR_RETURN(
        auto exe, base_client_->Compile(sharded_computation, sharded_options));
    TF_ASSIGN_OR_RETURN(auto program_shape, computation.GetProgramShape());
    fast_path_exe = std::make_unique<WrapperPjRtExecutable>(
        std::move(exe), this, std::move(program_shape));

    // currently no good mechanism for copying exe so just compile again
    // to create a "slow path" executable
    task.compiler->Compile(
        /*run_id=*/0,
        {.replica_count = num_replicas, .num_partitions = num_partitions});
  } else {
    absl::flat_hash_set<HloModule*> already_built;
    for (auto&& op : compile_result.schedule) {
      std::visit(
          overloaded{
              [&](const SpmdHloModuleTask& task) {
                if (already_built.contains(task.module->module.get())) {
                  return;
                }
                for (int64_t local_device = 0;
                     local_device < addressable_devices.size();
                     ++local_device) {
                  const int64_t global_device_id =
                      addressable_devices[local_device]
                          ->global_device_id()
                          .value();
                  if (task.device_assignment.Contains(global_device_id)) {
                    context_->CreateCompileTask(
                        addressable_devices[local_device]
                            ->local_device_id()
                            .value(),
                        task.compiler);
                    already_built.insert(task.module->module.get());
                  }
                }
              },
              [](const auto&) {}},
          op.op);
    }
  }

  // this is super annoying, but now necessary
  // the context surrounding the Python call has to be preserved which means
  // we have to block on all the compile tasks finishing
  // we also migth need information from the compiled XLA executable
  // to determine the launch
  context_->FenceCompilation();

  // wrap into PjRtLoadedExecutable
  return std::unique_ptr<PjRtLoadedExecutable>(
      std::make_unique<LegatePjRtExecutable>(
          this, base_client_.get(), computation.proto().name(),
          std::move(program_shape), std::move(compile_result.schedule),
          std::move(compile_result.temporaries), std::move(device_assignment),
          std::move(addressable_device_logical_ids),
          std::move(addressable_devices), compile_result.executable_layout,
          compile_result.parameter_layouts, compile_result.output_layouts,
          std::move(compile_result.parameter_shardings),
          std::vector<Shape>{compile_result.output_shape},
          std::move(compile_result.root_shardings), std::move(fast_path_exe)));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> LegateClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  XlaComputation xla_computation;

  // NOTE: we cannot allow return tuples as we don't support them in
  // LegateBuffer's atm
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, /*use_shardy=*/false));

  return Compile(xla_computation, options);
}

LegateClient::LegateClient(std::unique_ptr<PjRtClient> base_client,
                           Backend* backend,
                           std::shared_ptr<ZukuExecuteContext> context)
    : base_client_(base_client.release()),
      backend_(backend),
      transpose_cache_(1024),
      context_(std::move(context)) {}

LegateClient::~LegateClient() {
  // The streams must be cleared after shutting down Legate
  // to ensure that no tasks are still running that
  // require the streams
  ClearCachedStreams();
}

absl::Status CompileHloModuleFromFile(const std::string& hlo_file,
                                      const std::string& platform_name,
                                      int replica_count, int num_partitions,
                                      bool erase_sharding, bool auto_sharding,
                                      std::optional<int64_t> device_mem) {
  TF_ASSIGN_OR_RETURN(auto* platform, PlatformUtil::GetPlatform(platform_name));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Backend> backend,
      Backend::CreateBackend(BackendOptions().set_platform(platform)));

  TF_ASSIGN_OR_RETURN(
      auto hlo_module,
      LoadModuleFromFile(
          hlo_file, "pb", hlo_module_loader_details::Config(),
          [replica_count = replica_count,
           num_partitions = num_partitions](HloModuleConfig* config) {
            config->set_seed(42);
            config->set_replica_count(replica_count);
            config->set_num_partitions(num_partitions);
            if (num_partitions > 1) {
              config->set_use_spmd_partitioning(true);
            }
          }));

  if (erase_sharding) {
    for (auto* comp : hlo_module->computations()) {
      for (auto* instruction : comp->MakeInstructionPostOrder()) {
        instruction->clear_sharding();
      }
    }
  }

  HloModuleProto proto = hlo_module->ToProto();
  XlaComputation computation{proto};

  auto da = std::make_shared<DeviceAssignment>(replica_count, num_partitions);
  int dev = 0;
  for (int r = 0; r < replica_count; ++r) {
    for (int c = 0; c < num_partitions; ++c, ++dev) {
      (*da)(r, c) = dev;
    }
  }

  ExecutableBuildOptions exec_options;
  exec_options.set_num_partitions(num_partitions)
      .set_num_replicas(replica_count)
      .set_use_auto_spmd_partitioning(auto_sharding)
      .set_use_spmd_partitioning(true)
      .set_allow_spmd_sharding_propagation_to_output({true})
      .set_device_assignment(*da)
      .mutable_debug_options();
  CompileOptions options{.executable_build_options = std::move(exec_options)};

  TF_ASSIGN_OR_RETURN(auto compile_result,
                      CreateTasks(computation, options, da, backend.get(),
                                  /*client=*/nullptr, {}));

  for (auto&& op : compile_result.schedule) {
    std::visit(overloaded{[&](const SpmdHloModuleTask& task) {
                            LegateCompileConfig config{
                                .replica_count = replica_count,
                                .num_partitions = num_partitions,
                                .device_mem = device_mem};
                            task.compiler->Compile(0, config);
                          },
                          [](const auto&) {}},
               op.op);
  }
  return absl::OkStatus();
}

}  // namespace xla

extern "C" void CompileHloModuleFromFile(const std::string& hlo_file,
                                         const std::string& platform_name,
                                         int replica_count, int num_partitions,
                                         bool erase_sharding, bool autoshard,
                                         std::optional<int64_t> device_mem) {
  auto status = xla::CompileHloModuleFromFile(
      hlo_file, platform_name, replica_count, num_partitions, erase_sharding,
      autoshard, device_mem);
  if (!status.ok()) {
    std::cerr << status.message() << std::endl;
  }
}

extern "C" void EnableFastPath(bool enable) {
  xla::enable_fast_path_exe = enable;
}

extern "C" void ReplicateParametersSmallerThanNumElements(
    int64_t num_elements) {
  if (num_elements == 0) {
    xla::replicated_parameter_num_elements_cutoff = std::nullopt;
  } else {
    xla::replicated_parameter_num_elements_cutoff = num_elements;
  }
}

extern "C" void RecomputeArgumentsIfCostLessThan(int64_t cost) {
  if (cost == 0) {
    xla::recompute_from_arguments_if_cost_less_than = std::nullopt;
  } else {
    xla::recompute_from_arguments_if_cost_less_than = cost;
  }
}

extern "C" void EnableOnlyFuseLoopTasks(bool enable) {
  xla::only_fuse_loop_tasks = enable;
}

extern "C" void EnableTaskFusion(bool enable) { xla::use_task_fusion = enable; }
