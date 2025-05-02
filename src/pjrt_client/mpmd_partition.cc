/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_partition.h"

#include <optional>

#include "mpmd_reorder_shard_map_transpose.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/iota_sharding_sanitizer.h"
#include "xla/pjrt/multimesh/mm_sharding.h"
#include "xla/pjrt/multimesh/mpmd_argument_recompute.h"
#include "xla/pjrt/multimesh/mpmd_buffer_scheduling_name.h"
#include "xla/pjrt/multimesh/mpmd_coloring.h"
#include "xla/pjrt/multimesh/mpmd_computation_fusion.h"
#include "xla/pjrt/multimesh/mpmd_computation_grouper.h"
#include "xla/pjrt/multimesh/mpmd_computation_inliner.h"
#include "xla/pjrt/multimesh/mpmd_concatenate_grouper.h"
#include "xla/pjrt/multimesh/mpmd_constant_output_copy.h"
#include "xla/pjrt/multimesh/mpmd_cross_task_barrier_remover.h"
#include "xla/pjrt/multimesh/mpmd_cut_size_minimizer.h"
#include "xla/pjrt/multimesh/mpmd_hoist_loop_convert.h"
#include "xla/pjrt/multimesh/mpmd_hoist_shard_map_reduce.h"
#include "xla/pjrt/multimesh/mpmd_input_output_buffer_alias.h"
#include "xla/pjrt/multimesh/mpmd_insert_reshard.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_instruction_delay_recolor.h"
#include "xla/pjrt/multimesh/mpmd_logical_sharding_propagation.h"
#include "xla/pjrt/multimesh/mpmd_logical_to_gspmd_sharding.h"
#include "xla/pjrt/multimesh/mpmd_loop_unroll.h"
#include "xla/pjrt/multimesh/mpmd_microbatch_loop_canonicalizer.h"
#include "xla/pjrt/multimesh/mpmd_parameter_replication.h"
#include "xla/pjrt/multimesh/mpmd_repeated_output_copy.h"
#include "xla/pjrt/multimesh/mpmd_shard_map_loop_reduce.h"
#include "xla/pjrt/multimesh/mpmd_sharding_propagation.h"
#include "xla/pjrt/multimesh/mpmd_simple_loop_increment_coloring.h"
#include "xla/pjrt/multimesh/mpmd_store.h"
#include "xla/pjrt/multimesh/mpmd_reorder_shard_map_transpose.h"
#include "xla/pjrt/multimesh/mpmd_uniquify_colors.h"
#include "xla/pjrt/multimesh/mpmd_unused_loop_output_remover.h"
#include "xla/pjrt/multimesh/mpmd_unused_param_output_remover.h"
#include "xla/pjrt/multimesh/scalar_argument.h"
#include "xla/service/call_inliner.h"
#include "xla/service/dump.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape.h"
#include "xla/util.h"

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

namespace xla {
namespace {

constexpr absl::string_view kHoistConvertEnv = "MULTIMESH_HOIST_CONVERT";
constexpr absl::string_view kZeroArgsEnv = "MULTIMESH_ZERO_ARGUMENTS";
constexpr absl::string_view kRemoveHoistedReduce =
    "MULTIMESH_REMOVE_HOISTED_REDUCE";
constexpr absl::string_view kLoopIncrementColorEnv =
    "MULTIMESH_LOOP_INCREMENT_COLOR";
constexpr absl::string_view kArgumentRecomputeEnv =
    "MULTIMESH_ARGUMENT_RECOMPUTE";
constexpr absl::string_view kCutSizeMinimizeEnv = "MULTIMESH_MINIMIZE_CUT_SIZE";
constexpr absl::string_view kColorPropagationPriorityEnv =
    "MULTIMESH_COLOR_PROPAGATION_PRIORITY";

absl::StatusOr<HloSharding> ToMpmdSharding(
    const Shape& shape, const HloSharding& iota_sharding,
    const zuku::DeviceList& task_devices,
    const zuku::DeviceList& global_devices, bool use_submesh_sharding) {
  if (iota_sharding.IsReplicated() && task_devices == global_devices) {
    return HloSharding::Replicate();
  }

  if (task_devices.size() == 1 &&
      (iota_sharding.IsTileMaximalLeaf() || iota_sharding.IsReplicated())) {
    return HloSharding::Replicate().AssignDevice(task_devices.start());
  }

  OpSharding proto = iota_sharding.ToProto();

  if (proto.iota_reshape_dims_size() > 0) {
    proto.set_iota_offset(task_devices.start());
  } else {
    for (size_t idx = 0; idx < proto.tile_assignment_devices_size(); ++idx) {
      int64_t relative_id = proto.tile_assignment_devices(idx);
      proto.set_tile_assignment_devices(idx, task_devices[relative_id]);
    }
  }

  if (iota_sharding.IsReplicated() && shape.dimensions_size() > 0) {
    // replicated on subset of devices
    proto.set_type(OpSharding::OTHER);
    for (const auto& dim : shape.dimensions()) {
      proto.add_tile_assignment_dimensions(1);
    }
    proto.add_tile_assignment_dimensions(task_devices.size());
    proto.mutable_iota_reshape_dims()->Add(task_devices.size());
    proto.mutable_iota_transpose_perm()->Add(0);
    proto.set_iota_offset(task_devices.start());
    proto.set_replicate_on_last_tile_dim(true);
  }

  return HloSharding::FromProto(proto);
}

absl::StatusOr<HloSharding> ToMpmdSharding(
    HloInstruction* instruction, const zuku::DeviceList& task_devices,
    const zuku::DeviceList& global_devices, bool use_submesh_sharding) {
  if (instruction->has_sharding()) {
    return ToMpmdSharding(instruction->shape(), instruction->sharding(),
                          task_devices, global_devices, use_submesh_sharding);
  }
  return ToMpmdSharding(instruction->shape(), HloSharding::Replicate(),
                        task_devices, global_devices, use_submesh_sharding);
}

// 0 is sentinel value indicating no temp offload
int64_t temp_offload_min_reuse_distance = 0ULL;

int64_t temp_offload_min_size = 1024 * 1024;

static constexpr std::array kSkipComputationNames = {"_threefry", "_normal_",
                                                     "_where", "silu"};

absl::StatusOr<HloModuleConfig> GetHloModuleConfig(
    HloComputation* entry_computation,
    const ExecutableBuildOptions& build_options) {
  ProgramShapeProto program_shape_proto;
  for (auto* parameter : entry_computation->parameter_instructions()) {
    program_shape_proto.mutable_parameters()->Add(parameter->shape().ToProto());
    program_shape_proto.mutable_parameter_names()->Add(
        std::string(parameter->name()));
    *program_shape_proto.mutable_result() =
        entry_computation->root_instruction()->shape().ToProto();
  }

  ProgramShape program_shape{program_shape_proto};

  return HloModule::CreateModuleConfigFromShape(std::move(program_shape),
                                                build_options.debug_options());
}

absl::StatusOr<std::unique_ptr<HloModuleConfig>> GetHloModuleConfig(
    const ProgramShape& program_shape,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options) {
  // Validate incoming layouts.
  if (argument_layouts.size() != program_shape.parameters_size()) {
    return InvalidArgument(
        "Invalid number of arguments for computation: expected %d, got %u.",
        program_shape.parameters_size(), argument_layouts.size());
  }
  build_options.debug_options();

  ExecutionOptions execution_options =
      CreateExecutionOptions(build_options, &program_shape);

  return CreateModuleConfig(program_shape, argument_layouts, &execution_options,
                            build_options.num_replicas());
}

// Helper function for generating an HLO module config from the
// standard input arguments `shape`, `argument_layouts`, and `build_options`
// that are passed to `PjRtClient::Compile`
absl::StatusOr<std::unique_ptr<HloModuleConfig>> GetHloModuleConfig(
    const HloModuleProto& proto,
    const absl::Span<const Shape* const> argument_layouts,
    const ExecutableBuildOptions& build_options) {
  TF_RET_CHECK(proto.has_host_program_shape());
  ProgramShape program_shape(proto.host_program_shape());
  return GetHloModuleConfig(program_shape, argument_layouts, build_options);
}

class MpmdScheduler {
 public:
  MpmdScheduler(HloModule* module, const HloPartition& partition)
      : module_(module),
        partition_(partition),
        default_replicated_(HloSharding::Replicate()),
        num_temp_stores_(0) {}

  absl::Status AddStore(const Store::Type type, HloInstruction* instruction,
                        std::optional<int64_t> index) {
    auto devices = [&]() -> absl::StatusOr<zuku::DeviceList> {
      auto color = Color(instruction);
      if (color.has_value()) {
        return partition_.DevicesForColor(*color);
      }

      if (instruction->opcode() == HloOpcode::kParameter) {
        return GetDevices(instruction->sharding_or_default(default_replicated_),
                          partition_.Devices());
      };
      return InvalidArgumentStrCat("instruction ", instruction->name(),
                                   " has no color");
    }();

    if (!devices.ok()) {
      return devices.status();
    }

    auto color = Color(instruction);
    TF_ASSIGN_OR_RETURN(zuku::ShardedShape sharded_shape,
                        XlaShapeToZukuShape(instruction->shape(), *devices,
                                            instruction->sharding_or_default(
                                                default_replicated_)));

    auto allow_sharding_override = [](int64_t index,
                                      absl::Span<const bool> allow) {
      if (allow.size() > index) {
        return allow[index];
      }
      if (allow.empty()) {
        return false;
      }
      return allow[0];
    };

    if (type == Store::Type::TEMP) {
      index = num_temp_stores_++;
    }

    const bool use_submesh_sharding = [&] {
      HloModule* module = instruction->parent()->parent();
      if (type == Store::Type::PARAM) {
        return allow_sharding_override(
            *index,
            module->config().allow_spmd_sharding_propagation_to_parameters());
      }
      if (type == Store::Type::ROOT) {
        return allow_sharding_override(
            *index,
            module->config().allow_spmd_sharding_propagation_to_output());
      }
      // always allow sharding overrides for temps
      return true;
    }();

    TF_ASSIGN_OR_RETURN(
        HloSharding mpmd_sharding,
        ToMpmdSharding(instruction, *devices, partition_.Devices(),
                       use_submesh_sharding));

    VLOG(5) << "Adding store for " << instruction->name() << ", type=" << type
            << ", index=" << *index << ", mpmd_sharding=" << mpmd_sharding
            << ", use_submesh_sharding=" << std::boolalpha
            << use_submesh_sharding << "";

    scheduling_name_to_buffer_index_.emplace(
        instruction->metadata().scheduling_name(),
        Store{type, *index, instruction->metadata().scheduling_name(),
              /*scalar=*/false, std::move(sharded_shape), mpmd_sharding,
              instruction->shape()});
    return absl::OkStatus();
  }

  std::vector<MpmdOperation> schedule() const { return schedule_; }

  std::vector<std::shared_ptr<SpmdModule>> modules() const {
    std::vector<std::shared_ptr<SpmdModule>> mods;
    for (const auto& [call, module] : modules_) {
      mods.push_back(module);
    }
    return mods;
  }

  bool AssignedToStore(HloInstruction* instruction) {
    return scheduling_name_to_buffer_index_.contains(
        instruction->metadata().scheduling_name());
  }

  absl::Status AddReshardTask(HloInstruction* reshard) {
    if (!already_scheduled_.contains(reshard)) {
      TF_ASSIGN_OR_RETURN(auto source_store,
                          GetStore(reshard->mutable_operand(0)));
      TF_ASSIGN_OR_RETURN(auto dest_store, GetStore(reshard));
      schedule_.push_back(
          MpmdOperation{.op = Reshard{.input = std::move(source_store),
                                      .output = std::move(dest_store)}});
      already_scheduled_.insert(reshard);
    }
    return absl::OkStatus();
  }

  absl::Status PrioritizeReshardsInSchedule(HloInstruction* source) {
    for (auto* user : source->users()) {
      if (user->IsCustomCall(kCustomCallReshard)) {
        TF_RETURN_IF_ERROR(AddReshardTask(user));
      }
    }
    return absl::OkStatus();
  }

  std::vector<Store> temporaries() const {
    std::vector<std::optional<Store>> temps(NumTemporaries());
    for (const auto& [name, store] : scheduling_name_to_buffer_index_) {
      if (store.type == Store::Type::TEMP) {
        temps[store.index] = store;
      }
    }

    std::vector<Store> stores;
    stores.reserve(temps.size());
    for (const auto& temp : temps) {
      stores.push_back(*std::move(temp));
    }
    return stores;
  }

  int64_t NumTemporaries() const { return num_temp_stores_; }

  absl::Status AddHloModuleTask(HloInstruction* call,
                                const ExecutableBuildOptions& options);

  void LogUseOrder(const HloInstruction* user, const HloInstruction* operand) {
    last_use_[operand] = user;
  }

  absl::StatusOr<Store> GetStore(const HloInstruction* instruction) {
    auto iter = scheduling_name_to_buffer_index_.find(
        instruction->metadata().scheduling_name());
    if (iter == scheduling_name_to_buffer_index_.end()) {
      return InvalidArgumentStrCat(
          instruction->name(), " with scheduling name ",
          instruction->metadata().scheduling_name(), " has no assigned store");
    }
    return iter->second;
  }

 private:
  HloModule* module_;
  HloSharding default_replicated_;
  const HloPartition& partition_;
  absl::flat_hash_set<HloInstruction*> already_scheduled_;
  std::vector<MpmdOperation> schedule_;
  absl::flat_hash_map<std::string, Store> scheduling_name_to_buffer_index_;
  absl::flat_hash_map<const HloInstruction*, const HloInstruction*> last_use_;
  absl::flat_hash_map<const HloComputation*, std::shared_ptr<SpmdModule>>
      modules_;
  int64_t num_temp_stores_;
};

absl::Status MpmdScheduler::AddHloModuleTask(
    HloInstruction* call, const ExecutableBuildOptions& options) {
  std::vector<Store> inputs;
  std::vector<Store> outputs;
  std::vector<ScalarArgument> scalars;
  absl::flat_hash_set<int64_t> temporary_param_indices;
  absl::flat_hash_set<int64_t> temporary_root_indices;
  absl::flat_hash_map<std::string, int64_t> input_names;
  absl::flat_hash_map<int64_t, int64_t> in_out_name_alias;
  HloComputation* computation = call->called_computations()[0];
  std::optional<int64_t> slice_param_number{};

  auto color = Color(call);
  if (!color.has_value()) {
    return InvalidArgumentStrCat("call instruction ", call->name(),
                                 " has no assigned color");
  }
  auto devices = partition_.DevicesForColor(*color);

  for (int64_t index = 0; index < call->operand_count(); ++index) {
    HloInstruction* operand = call->mutable_operand(index);

    if (operand->IsCustomCall(kCustomCallSliceOffset)) {
      TF_ASSIGN_OR_RETURN(int offset, GetAttribute<int>(operand, "offset"));
      scalars.push_back(ScalarArgument{offset, index});
      VLOG(3) << "pushing back offset " << offset << " for task "
              << call->called_computations()[0]->name();

      inputs.push_back(Store{.type = Store::Type::TEMP,
                             .index = -1,
                             .name = "slice-offset",
                             .scalar = true,
                             .mpmd_sharding = HloSharding::Replicate(),
                             .shape = operand->shape()});
      slice_param_number = index;
    } else {
      TF_ASSIGN_OR_RETURN(Store input, GetStore(operand));
      VLOG(5) << call->called_computations()[0]->name() << " adding input "
              << index << " " << operand->name() << ", type=" << input.type
              << ", index=" << input.index << " with scheduling name "
              << operand->metadata().scheduling_name();
      input_names[operand->metadata().scheduling_name()] = index;
      // need to track these for input/output alias pass later
      if (input.type == Store::Type::TEMP) {
        temporary_param_indices.insert(index);
      }
      inputs.push_back(std::move(input));
    }
  }

  std::vector<HloInstruction*> sorted_users(call->shape().tuple_shapes_size(),
                                            nullptr);
  for (auto* user : call->users()) {
    sorted_users[user->tuple_index()] = user;
  }

  for (int64_t index = 0; index < sorted_users.size(); ++index) {
    HloInstruction* user = sorted_users[index];
    TF_ASSIGN_OR_RETURN(Store output, GetStore(user));
    VLOG(5) << call->called_computations()[0]->name() << " adding output "
            << user->tuple_index() << " " << user->name()
            << ", type=" << output.type << ", index=" << output.index
            << " with scheduling name " << user->metadata().scheduling_name();
    // need to track these for input/output alias pass later
    if (output.type == Store::Type::TEMP) {
      temporary_root_indices.insert(user->tuple_index());
    }
    outputs.push_back(std::move(output));
    auto alias_iter = input_names.find(user->metadata().scheduling_name());
    if (alias_iter != input_names.end()) {
      VLOG(5) << "adding loop-carried alias pair " << alias_iter->second << ": "
              << user->tuple_index() << " for buffer name "
              << user->metadata().scheduling_name() << " for user "
              << user->name() << " of " << call->name();
      in_out_name_alias[alias_iter->second] = user->tuple_index();
    }
  }

  // no reason to include a dummy task generating the slice offset
  if (outputs.size() == 1 && outputs.front().scalar) {
    return absl::OkStatus();
  }

  auto add_module = [&](std::shared_ptr<SpmdModule> module) {
    const bool is_loop = absl::StrContains(module->module->name(), "loop");
    SpmdHloModuleTask task{.module = std::move(module),
                           .device_assignment = devices,
                           .inputs = std::move(inputs),
                           .outputs = std::move(outputs),
                           .scalars = std::move(scalars),
                           .loop = is_loop};
    schedule_.push_back({std::move(task)});
    return absl::OkStatus();
  };

  auto iter = modules_.find(call->called_computations()[0]);
  if (iter != modules_.end()) {
    return add_module(iter->second);
  }

  HloInputOutputAliasConfig in_out_alias_config{
      computation->root_instruction()->shape()};
  HloBufferDonorConfig donor_config;

  for (int64_t input_number = 0; input_number < computation->num_parameters();
       ++input_number) {
    const auto& input_store = inputs[input_number];
    HloInstruction* operand = call->mutable_operand(input_number);
    if (input_store.type == Store::Type::PARAM && last_use_[operand] == call) {
      auto output_index = module_->input_output_alias_config().GetAliasedOutput(
          input_store.index,

          {});
      if (output_index.has_value()) {
        const int64_t root_number = [&] {
          if (output_index->empty()) {
            return int64_t(0);
          }
          return output_index->front();
        }();
        int output_number = 0;
        for (const auto& output_store : outputs) {
          if (output_store.type == Store::Type::ROOT &&
              output_store.index == root_number) {
            VLOG(5) << "mapping param/root alias " << input_store.index << ":"
                    << root_number << " to " << input_number << ":"
                    << output_number;
            TF_RETURN_IF_ERROR(in_out_alias_config.SetUpAlias(
                {output_number}, input_number, {}));
          }
          ++output_number;
        }
      } else if (module_->buffer_donor_config().ParameterIsBufferDonor(
                     input_store.index, {}) &&
                 ShapeUtil::ElementsIn(operand->shape()) > 1) {
        // TODO: enable buffer donation for scalars once scalar broadcasts
        // are more stable in zuku
        TF_RETURN_IF_ERROR(donor_config.AddBufferDonor(input_number, {}));
      }
    }
  }

  for (const auto [param_number, output_index] : in_out_name_alias) {
    VLOG(5) << "adding loop-carried alias pair " << param_number << ": "
            << output_index;
    TF_RETURN_IF_ERROR(
        in_out_alias_config.SetUpAlias({output_index}, param_number, {}));
  }

  std::vector<const Shape*> layouts;
  layouts.reserve(inputs.size());

  ProgramShapeProto program_shape_proto;
  for (auto* parameter : computation->parameter_instructions()) {
    program_shape_proto.mutable_parameters()->Add(parameter->shape().ToProto());
    program_shape_proto.mutable_parameter_names()->Add(
        std::string(parameter->name()));
    *program_shape_proto.mutable_result() =
        computation->root_instruction()->shape().ToProto();
  }

  ProgramShape program_shape{program_shape_proto};
  TF_ASSIGN_OR_RETURN(auto submodule_config,
                      GetHloModuleConfig(computation, options));
  auto submodule = std::make_unique<HloModule>(std::string(computation->name()),
                                               std::move(submodule_config));
  HloCloneContext clone_context{submodule.get()};
  submodule->AddComputationAndUnifyNamesAndIds(
      computation->CloneInContext(clone_context),
      /*is_entry=*/true);

  submodule->set_buffer_donor_config(std::move(donor_config));
  submodule->set_input_output_alias_config(std::move(in_out_alias_config));

  auto* root = submodule->entry_computation()->root_instruction();
  if (!root->has_sharding() && root->shape().IsTuple()) {
    // I haven't figured out the shape tree constructor yet
    OpSharding op_sharding;
    op_sharding.set_type(OpSharding::TUPLE);
    for (auto* operand : root->operands()) {
      if (operand->has_sharding()) {
        *op_sharding.add_tuple_shardings() = operand->sharding().ToProto();
      } else {
        *op_sharding.add_tuple_shardings();
      }
    }
    TF_ASSIGN_OR_RETURN(auto tuple_sharding,
                        HloSharding::FromProto(op_sharding));
    root->set_sharding(std::move(tuple_sharding));
  }

  DumpHloModuleIfEnabled(*submodule, "after_mpmd_partitioning");

  MpmdInputOutputBufferAlias aliaser{temporary_param_indices,
                                     temporary_root_indices};
  TF_ASSIGN_OR_RETURN(bool alias_changed, aliaser.Run(submodule.get()));

  // this will have assigned all the alias pairs
  // at this point, clear the buffer donors so that new aliases
  // don't get add after this point
  submodule->set_buffer_donor_config({});

  // at this point, sharding has been propagated to outputs
  // don't allow propagating again to write over them
  submodule->mutable_config().set_allow_spmd_sharding_propagation_to_parameters(
      {false});
  // submodule->mutable_config()
  //   .set_allow_spmd_sharding_propagation_to_outputs({true});
  submodule->mutable_config().set_num_partitions(devices.size());
  submodule->mutable_config().set_use_spmd_partitioning(true);

  LOG(INFO) << module_->name() << " partitioned into subtask "
            << submodule->name() << " on devices=[" << devices.start() << "..."
            << devices.stop() << "]"
            << ", module has " << submodule->instruction_count()
            << " instructions";

  auto spmd_module_ptr = std::make_shared<SpmdModule>(SpmdModule{
      .module = std::move(submodule),
      .loop_increment = partition_.IsLoopIncrementColor(*color),
      .slice_param_number = slice_param_number,
  });

  modules_[call->called_computations()[0]] = spmd_module_ptr;

  return add_module(std::move(spmd_module_ptr));
}

std::optional<int64_t> GetRootIndex(HloInstruction* instruction) {
  HloInstruction* root = instruction->parent()->root_instruction();
  if (root->opcode() != HloOpcode::kTuple) {
    if (instruction == root) {
      return 0;
    }
    return std::nullopt;
  }

  // if there are only a few users, first filter based on whether
  // the root is a known user
  if (instruction->user_count() <= 4) {
    const bool is_root =
        absl::c_any_of(instruction->users(),
                       [&](const HloInstruction* i) { return i == root; });
    if (!is_root) {
      return std::nullopt;
    }
  }

  // if this is (probably) a root, find the operand index
  for (int64_t index = 0; index < root->operand_count(); ++index) {
    if (root->operand(index) == instruction) {
      return index;
    }
  }
  return std::nullopt;
}

absl::Status AddTasksFromEntryComputation(
    MpmdScheduler& scheduler, HloComputation* computation, HloModule* module,
    const HloPartition& partition, const ExecutableBuildOptions& options,
    const InstructionProperties& properties) {
  auto postorder = computation->MakeInstructionPostOrder();
  // first configure all the parameters and roots
  for (auto* instruction : postorder) {
    if (instruction->opcode() == HloOpcode::kParameter) {
      TF_RETURN_IF_ERROR(scheduler.AddStore(Store::Type::PARAM, instruction,
                                            instruction->parameter_number()));
    } else if (instruction->opcode() == HloOpcode::kGetTupleElement ||
               instruction->IsCustomCall(kCustomCallReshard)) {
      std::optional<int64_t> root_index = GetRootIndex(instruction);
      if (root_index.has_value()) {
        TF_RETURN_IF_ERROR(
            scheduler.AddStore(Store::Type::ROOT, instruction, *root_index));
      }
    }
  }

  for (auto* instruction : postorder) {
    if (instruction->opcode() == HloOpcode::kCall ||
        instruction->IsCustomCall(kCustomCallReshard)) {
      for (auto* operand : instruction->operands()) {
        scheduler.LogUseOrder(instruction, operand);
      }
    }
  }

  // now go back through and configure all the temporaries
  for (auto* instruction : postorder) {
    if (instruction->opcode() == HloOpcode::kGetTupleElement ||
        instruction->IsCustomCall(kCustomCallReshard)) {
      if (!scheduler.AssignedToStore(instruction)) {
        TF_RETURN_IF_ERROR(
            scheduler.AddStore(Store::Type::TEMP, instruction, std::nullopt));
      }
    }
  }

  // at this point, every instruction should have been assigned to a unique
  // Store
  for (auto* instruction : postorder) {
    if (instruction->opcode() == HloOpcode::kCall) {
      TF_RETURN_IF_ERROR(scheduler.AddHloModuleTask(instruction, options));
    } else if (instruction->opcode() == HloOpcode::kGetTupleElement) {
      TF_RETURN_IF_ERROR(scheduler.PrioritizeReshardsInSchedule(instruction));
    } else if (instruction->IsCustomCall(kCustomCallReshard)) {
      TF_RETURN_IF_ERROR(scheduler.AddReshardTask(instruction));
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::tuple<std::vector<MpmdOperation>,
                          std::vector<std::shared_ptr<SpmdModule>>,
                          std::vector<Store>, std::vector<Store>>>
MpmdPartitionIntoTasks(HloModule* module, const HloPartition& partition,
                       const ExecutableBuildOptions& options) {
  auto properties = InstructionProperties::Create(module);
  MpmdScheduler scheduler{module, partition};

  // at this point, everything should have been flattened into the entry
  // computation
  TF_RETURN_IF_ERROR(
      AddTasksFromEntryComputation(scheduler, module->entry_computation(),
                                   module, partition, options, properties));

  std::vector<MpmdOperation> schedule = scheduler.schedule();
  std::vector<std::shared_ptr<SpmdModule>> modules = scheduler.modules();
  std::vector<Store> temporaries = scheduler.temporaries();

  std::vector<Store> unused_parameters;
  for (auto* param : module->entry_computation()->parameter_instructions()) {
    if (param->user_count() == 0) {
      TF_ASSIGN_OR_RETURN(auto store, scheduler.GetStore(param));
      unused_parameters.push_back(std::move(store));
    }
  }

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "Unique tasks";
    absl::flat_hash_set<SpmdModule*> visited;
    for (auto& op : schedule) {
      std::visit(overloaded{[&](const SpmdHloModuleTask& task) {
                              if (!visited.contains(task.module.get())) {
                                VLOG(3)
                                    << task.module->module->name()
                                    << ", devices=["
                                    << task.device_assignment.start() << "..."
                                    << task.device_assignment.stop() << "]";
                                visited.insert(task.module.get());
                              }
                            },
                            [&](const auto& unwrapped) {}},
                 op.op);
    }
  }

  if (VLOG_IS_ON(5)) {
    VLOG(5) << "Task order";
    for (auto& op : schedule) {
      std::visit(overloaded{[&](const SpmdHloModuleTask& task) {
                              VLOG(5) << task.module->module->name()
                                      << " on devices=["
                                      << task.device_assignment.start() << "..."
                                      << task.device_assignment.stop() << "]";
                            },
                            [&](const auto& unwrapped) {}},
                 op.op);
    }
  }

  return std::make_tuple(std::move(schedule), std::move(modules),
                         std::move(temporaries), std::move(unused_parameters));
}

absl::StatusOr<std::tuple<std::vector<MpmdOperation>,
                          std::vector<std::shared_ptr<SpmdModule>>,
                          std::vector<Store>, std::vector<Store>>>
MpmdPartition(const HloModuleProto& proto, const CompileOptions& options,
              const std::vector<const Shape*>& argument_layout_pointers,
              const Shape& executable_layout,
              const MpmdPartitionConfig& config) {
  ExecutableBuildOptions exec_options = options.executable_build_options;

  LOG(INFO) << "MPMD Partitoning " << proto.name();

  // create module_config based on layout and build options
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> module_config,
                      GetHloModuleConfig(proto, argument_layout_pointers,
                                         options.executable_build_options));

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      CreateModuleFromProto(proto, *module_config,
                                            exec_options.run_backend_only()));

  if (DumpingEnabledForHloModule(proto.name(), exec_options.debug_options())) {
    DumpHloModuleIfEnabled(*module, "before_mpmd_preprocess");
  }

  for (auto* computation : module->computations()) {
    for (auto&& name : kSkipComputationNames) {
      if (absl::StrContains(computation->name(), name)) {
        for (auto&& instruction : computation->instructions()) {
          instruction->set_metadata_op_name("");
        }
        break;
      }
    }
  }

  // We want the entire HLO to be "flat" so we need to run the call inliner
  HloPassPipeline pre_mpmd_pipeline("mpmd-preprocess");
  pre_mpmd_pipeline.AddPass<HloDCE>();
  pre_mpmd_pipeline.AddPass<CallInliner>();
  pre_mpmd_pipeline.AddPass<TupleSimplifier>();
  pre_mpmd_pipeline.AddPass<HloDCE>();
  TF_ASSIGN_OR_RETURN(bool changed, pre_mpmd_pipeline.Run(module.get()));

  if (DumpingEnabledForHloModule(proto.name(), exec_options.debug_options())) {
    DumpHloModuleIfEnabled(*module, "after_mpmd_preprocess");
  }

  if (!options.executable_build_options.has_device_assignment()) {
    return InvalidArgumentStrCat("ExecutableBuildOptions for ", proto.name(),
                                 " does not have a device assignment");
  }

  const bool hoist_convert =
      GetEnvOption(kHoistConvertEnv, config.hoist_loop_convert);
  const bool zero_out_arguments = GetEnvOption(kZeroArgsEnv, false);
  const bool loop_increment_color = GetEnvOption(kLoopIncrementColorEnv, true);
  const bool recompute_arguments = GetEnvOption(kArgumentRecomputeEnv, true);
  const bool minimize_cut_size = GetEnvOption(kCutSizeMinimizeEnv, true);
  const bool remove_hoisted_reduces = GetEnvOption(
      kRemoveHoistedReduce, /*deflt=*/config.remove_hoisted_reduces);

  const MpmdColoring::ColorPropagationPriority color_propagation_priority =
      MpmdColoring::GetColorPropagationPriorityFromString(
          GetEnvOption(kColorPropagationPriorityEnv,
                       /*deflt=*/absl::string_view("WEIGHT")));

  std::vector<int64_t> devices = {
      options.executable_build_options.device_assignment().begin(),
      options.executable_build_options.device_assignment().end()};
  TF_ASSIGN_OR_RETURN(auto dl, CreateDeviceList(devices));

  TF_ASSIGN_OR_RETURN(HloPartition partition,
                      HloPartition::Create(module.get(), std::move(dl)));

  /* The module has two forms. Flat/inlined and grouped. In the flat/inlined
     form, all tasks are part of the same computation and the entire module is a
     flat entry computation. Tasks are distinguish by the mpmd_metadata `color`
     attribute. In the grouped form, distinct tasks are grouped into called
     computations. The `MpmdComputationInliner` reverses the effect of
     `MpmdComputationGrouper`.
  */
  HloPassPipeline mpmd_pipeline("mpmd-pipeline");
  mpmd_pipeline.AddPass<MpmdConstantOutputCopy>(&partition);
  mpmd_pipeline.AddPass<MpmdRepeatedOutputCopy>(&partition);
  mpmd_pipeline.AddPass<MpmdMicrobatchLoopCanonicalizer>(&partition);
  if (loop_increment_color) {
    mpmd_pipeline.AddPass<MpmdSimpleLoopIncrementColoring>(&partition);
  }
  mpmd_pipeline.AddPass<MpmdColoring>(&partition, color_propagation_priority);
  mpmd_pipeline.AddPass<MpmdConstantOutputCopy>(&partition);
  mpmd_pipeline.AddPass<MpmdUnusedLoopOutputRemover>(&partition);
  mpmd_pipeline.AddPass<MpmdConcatenateGrouper>(&partition);
  mpmd_pipeline.AddPass<MpmdLogicalShardingPropagation>();
  if (hoist_convert) {
    mpmd_pipeline.AddPass<MpmdHoistLoopConvert>(&partition, zero_out_arguments);
  }
  if (config.replicated_parameter_num_elements_cutoff.has_value()) {
    mpmd_pipeline.AddPass<MpmdParameterReplication>(
        &partition, *config.replicated_parameter_num_elements_cutoff);
  }
  mpmd_pipeline.AddPass<MpmdComputationGrouper>(&partition);
  mpmd_pipeline.AddPass<HloDCE>();
  mpmd_pipeline.AddPass<MpmdCrossTaskBarrierRemover>(
      &partition, /*remove_parameters=*/false);
  mpmd_pipeline.AddPass<MpmdLogicalToGSPMDSharding>(&partition);
  mpmd_pipeline.AddPass<IotaShardingSanitizer>(&partition);

  MpmdShardingPropagationConfig shard_cfg{
      .min_shard_override_size =
          config.replicated_parameter_num_elements_cutoff};
  mpmd_pipeline.AddPass<MpmdShardingPropagation>(
      &partition, MpmdShardingPropagation::PropagationMode::ForwardInputOutput,
      shard_cfg);
  mpmd_pipeline.AddPass<MpmdShardingPropagation>(
      &partition, MpmdShardingPropagation::PropagationMode::BackwardInputOutput,
      shard_cfg);
  mpmd_pipeline.AddPass<MpmdShardingPropagation>(
      &partition, MpmdShardingPropagation::PropagationMode::ForwardFull,
      shard_cfg);
  mpmd_pipeline.AddPass<MpmdComputationFusion>(
      &partition, MpmdComputationFusion::FusionType::kMatchingColor,
      /*only_fuse_loop_tasks=*/false);
  mpmd_pipeline.AddPass<HloDCE>();
  mpmd_pipeline.AddPass<MpmdUniquifyColors>(&partition);
  mpmd_pipeline.AddPass<MpmdInstructionDelayRecolor>(&partition);
  mpmd_pipeline.AddPass<MpmdComputationInliner>(&partition);
  if (recompute_arguments &&
      config.recompute_from_arguments_if_cost_less_than.has_value()) {
    mpmd_pipeline.AddPass<MpmdArgumentRecompute>(
        &partition, *config.recompute_from_arguments_if_cost_less_than);
  }
  if (minimize_cut_size) {
    mpmd_pipeline.AddPass<MpmdCutSizeMinimizer>(&partition);
  }
  mpmd_pipeline.AddPass<MpmdShardMapLoopReduce>(&partition);
  mpmd_pipeline.AddPass<MpmdHoistShardMapReduce>(&partition,
                                                 remove_hoisted_reduces);
  mpmd_pipeline.AddPass<MpmdReorderShardMapTranspose>();
  mpmd_pipeline.AddPass<MpmdComputationGrouper>(&partition);
  mpmd_pipeline.AddPass<MpmdCrossTaskBarrierRemover>(
      &partition, /*remove_parameters=*/true);
  mpmd_pipeline.AddPass<MpmdUnusedParamOutputRemover>(&partition);
  mpmd_pipeline.AddPass<MpmdComputationFusion>(
      &partition, MpmdComputationFusion::FusionType::kMatchingColor,
      /*only_fuse_loop_tasks=*/false);
  mpmd_pipeline.AddPass<MpmdComputationFusion>(
      &partition, MpmdComputationFusion::FusionType::kMatchingDevices,
      /*only_fuse_loop_tasks=*/true);
  mpmd_pipeline.AddPass<HloDCE>();
  mpmd_pipeline.AddPass<MpmdLoopUnroll>(&partition);
  mpmd_pipeline.AddPass<MpmdInsertReshard>(&partition);
  mpmd_pipeline.AddPass<MpmdAssignBufferSchedulingName>(&partition);
  TF_ASSIGN_OR_RETURN(bool mpmd_changed, mpmd_pipeline.Run(module.get()));

  DumpHloModuleIfEnabled(*module, "global_mpmd");

  return MpmdPartitionIntoTasks(module.get(), partition,
                                options.executable_build_options);
}

}  // namespace xla

extern "C" void SetHostOffloadMinReuseDistance(int64_t reuse_distance) {
  xla::temp_offload_min_reuse_distance = reuse_distance;
}

extern "C" void SetHostOffloadMinSize(int64_t size) {
  xla::temp_offload_min_size = size;
}
