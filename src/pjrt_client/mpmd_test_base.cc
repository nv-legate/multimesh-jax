/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_test_base.h"

#include <fstream>

#include "xla/hlo/builder//xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_partition.h"
#include "xla/pjrt/legate/mpmd_utils.h"
#include "xla/util.h"

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

namespace xla {
namespace {

DeviceAssignment GetDeviceAssignment(int num_devices) {
  DeviceAssignment da{1, num_devices};
  for (int d = 0; d < num_devices; ++d) {
    da(0, d) = d;
  }
  return da;
}

}  // namespace

MpmdTestBase::MpmdTestBase() : HloTestBase() {
  std::filesystem::path this_file = __FILE__;
  testdata_root_ = this_file.parent_path() / "testdata";
}

void MpmdTestBase::SetUp() {
  HloTestBase::SetUp();
  EnableLegateRecomputation(false);
}

void MpmdTestBase::TearDown() {
  HloTestBase::TearDown();
  ClearMetadataNameTasks();
}

absl::Status MpmdTestBase::VerifyHloModule(const HloModule& module,
                                           const HloPartition& partition) {
  for (auto* computation : module.computations()) {
    for (auto* instruction : computation->instructions()) {
      if (!IsReplicatedOrNotSharded(instruction) &&
          !instruction->sharding().IsTuple()) {
        const int64_t expected_num_devices =
            partition.NumDevicesForInstruction(instruction);
        if (instruction->sharding().tile_assignment().num_elements() !=
            expected_num_devices) {
          return InvalidArgumentStrCat(
              instruction->name(), " has sharding ",
              instruction->sharding().ToString(),
              " which has the wrong no. devices, expected ",
              expected_num_devices);
        }
      }
    }
  }
  return absl::OkStatus();
}

std::vector<HloInstruction*> MpmdTestBase::EntryComputationCalls(
    HloModule* module) {
  std::vector<HloInstruction*> calls;
  for (auto* instruction : module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      calls.push_back(instruction);
    }
  }
  return calls;
}

std::vector<HloComputation*> MpmdTestBase::EntryComputationCalledComputations(
    HloModule* module) {
  std::vector<HloComputation*> calls;
  for (auto* instruction : module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      calls.push_back(instruction->called_computations()[0]);
    }
  }
  return calls;
}

absl::StatusOr<HloSharding> MpmdTestBase::GetSharding(absl::string_view pbtxt) {
  OpSharding sharding_proto;
  if (!tsl::protobuf::TextFormat::ParseFromString(std::string(pbtxt),
                                                  &sharding_proto)) {
    return InvalidArgumentStrCat("failed to parse sharding proto");
  }
  return HloSharding::FromProto(sharding_proto);
}

absl::StatusOr<std::unique_ptr<HloModule>> MpmdTestBase::GetHloModuleFromText(
    absl::string_view hlo_text, int num_devices) {
  TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(hlo_text));

  zuku::DeviceList devices{{.start = 0, .num_devices = num_devices}};
  TF_ASSIGN_OR_RETURN(auto partition,
                      HloPartition::Create(module.get(), devices));
  partition_ = std::make_unique<HloPartition>(std::move(partition));

  if (num_devices > 1) {
    module->mutable_config().set_num_partitions(num_devices);
    module->mutable_config().set_use_spmd_partitioning(true);
  }

  return std::move(module);
}

absl::StatusOr<std::unique_ptr<HloModule>> MpmdTestBase::GetHloModuleFromPath(
    absl::string_view hlo_text_path, int num_devices) {
  auto full_path = testdata_root_ / hlo_text_path;
  std::ifstream ifs(full_path);
  if (ifs) {
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return GetHloModuleFromText(buffer.str(), num_devices);
  }
  return InvalidArgumentStrCat("test file ", hlo_text_path, "->",
                               full_path.string(), " does not exist");
}

absl::StatusOr<std::pair<std::vector<SpmdHloModuleTask>, int64_t>>
MpmdTestBase::RunMpmdOnHloModule(std::unique_ptr<HloModule> module,
                                 int num_devices, MpmdTestConfig cfg) {
  HloModuleProto proto = module->ToProto();

  auto* root = module->entry_computation()->root_instruction();
  size_t num_outputs =
      root->shape().IsTuple() ? root->shape().tuple_shapes_size() : 1;

  absl::Span<const bool> allow_sharding_propagation =
      module->config().allow_spmd_sharding_propagation_to_output();
  bool allow[] = {true};
  if (!cfg.use_module_config_auto_output_sharding) {
    // if not specified, set to
    // true for all outputs
    allow_sharding_propagation = allow;
  }

  CompileOptions options{
      .executable_build_options =
          ExecutableBuildOptions()
              .set_allow_spmd_sharding_propagation_to_output(
                  allow_sharding_propagation)
              .set_use_auto_spmd_partitioning(cfg.use_auto_input_sharding)};

  options.executable_build_options.set_device_assignment(
      GetDeviceAssignment(num_devices));

  // force allocation of the debug options
  options.executable_build_options.mutable_debug_options();

  XlaComputation comp(std::move(proto));

  TF_ASSIGN_OR_RETURN(auto program_shape, comp.GetProgramShape());
  std::vector<Shape> argument_layouts = program_shape.parameters();
  std::vector<const Shape*> argument_layout_pointers;
  argument_layout_pointers.reserve(argument_layouts.size());
  for (auto& shape : argument_layouts) {
    argument_layout_pointers.push_back(&shape);
  }

  const HloComputationProto* entry_comp = nullptr;
  for (const auto& c : comp.proto().computations()) {
    if (c.id() == comp.proto().entry_computation_id()) {
      entry_comp = &c;
      break;
    }
  }
  Shape result_shape(entry_comp->program_shape().result());

  auto result = MpmdPartition(
      comp.proto(), options, argument_layout_pointers, result_shape,
      {.replicated_parameter_num_elements_cutoff =
           cfg.replicated_parameter_num_elements_cutoff,
       .recompute_from_arguments_if_cost_less_than =
           cfg.recompute_from_arguments_if_cost_less_than,
       .run_simplification_passes = cfg.run_simplification_passes,
       .only_fuse_loop_tasks = cfg.only_fuse_loop_tasks});

  auto [schedule, modules, temporaries, unused_params] = *std::move(result);

  std::vector<SpmdHloModuleTask> tasks;
  for (auto& op : schedule) {
    std::visit(overloaded{[&](const SpmdHloModuleTask& task) {
                            tasks.push_back(task);
                          },
                          [&](const auto&) {}},
               op.op);
  }
  return std::make_pair(std::move(tasks), temporaries.size());
}

absl::StatusOr<std::pair<std::vector<SpmdHloModuleTask>, int64_t>>
MpmdTestBase::RunMpmdOnHloTextPath(absl::string_view hlo_text_path,
                                   int num_devices, MpmdTestConfig cfg) {
  TF_ASSIGN_OR_RETURN(auto module,
                      GetHloModuleFromPath(hlo_text_path, num_devices));
  return RunMpmdOnHloModule(std::move(module), num_devices, cfg);
}

absl::StatusOr<std::pair<std::vector<SpmdHloModuleTask>, int64_t>>
MpmdTestBase::RunMpmdOnHloString(absl::string_view hlo_string, int num_devices,
                                 MpmdTestConfig cfg) {
  TF_ASSIGN_OR_RETURN(auto module,
                      GetHloModuleFromText(hlo_string, num_devices));
  return RunMpmdOnHloModule(std::move(module), num_devices, cfg);
}

bool MpmdHloInstructionMatcher::MatchAndExplain(
    const HloInstruction* instruction,
    ::testing::MatchResultListener* listener) const {
  return match_and_explain_(instruction, listener);
}

bool MpmdTaskMatcher::MatchAndExplain(
    const SpmdHloModuleTask& task,
    ::testing::MatchResultListener* listener) const {
  return match_and_explain_(task, listener);
}

bool MpmdStoreMatcher::MatchAndExplain(
    const Store& store, ::testing::MatchResultListener* listener) const {
  return match_and_explain_(store, listener);
}

namespace mpmd_matchers {

std::vector<std::vector<HloInstruction*>> CalledComputationInstructions(
    HloModule* module) {
  std::vector<std::vector<HloInstruction*>> instruction_groups;
  std::vector<HloComputation*> to_visit{module->entry_computation()};
  while (!to_visit.empty()) {
    HloComputation* comp = to_visit.back();
    to_visit.pop_back();

    for (auto* instruction : comp->MakeInstructionPostOrder()) {
      switch (instruction->opcode()) {
        case HloOpcode::kWhile:
        case HloOpcode::kCall: {
          instruction_groups.push_back(instruction->called_computations()[0]
                                           ->MakeInstructionPostOrder());
          to_visit.push_back(instruction->called_computations()[0]);
          break;
        }
        default:
          break;
      }
    }
  }
  return instruction_groups;
}

std::vector<HloInstruction*> EntryComputationCalls(HloModule* module) {
  std::vector<HloInstruction*> calls;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      calls.push_back(instruction);
    }
  }
  return calls;
}

std::vector<HloInstruction*> FlatInstructions(HloModule* module) {
  std::vector<HloInstruction*> instructions;
  instructions.reserve(module->instruction_count());
  for (auto* comp : module->computations()) {
    for (auto* instruction : comp->instructions()) {
      instructions.push_back(instruction);
    }
  }
  return instructions;
}

std::vector<HloInstruction*> FlatInstructionsWithoutReducesAndPredicates(
    HloModule* module) {
  std::vector<HloInstruction*> instructions;
  instructions.reserve(module->instruction_count());
  std::vector<HloComputation*> to_visit{module->entry_computation()};
  while (!to_visit.empty()) {
    HloComputation* next = to_visit.back();
    to_visit.pop_back();
    for (auto* instruction : next->instructions()) {
      instructions.push_back(instruction);
      switch (instruction->opcode()) {
        case HloOpcode::kReduce:
          break;
        case HloOpcode::kWhile:
          to_visit.push_back(instruction->called_computations()[0]);
          break;
        default: {
          for (auto* comp : instruction->called_computations()) {
            to_visit.push_back(comp);
          }
          break;
        }
      }
    }
  }
  return instructions;
}

std::vector<HloComputation*> WhileBodies(HloModule* module) {
  std::vector<HloComputation*> computations;
  for (auto* instruction : module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      computations.push_back(instruction->called_computations()[0]);
    }
  }
  return computations;
}

::testing::Matcher<const ::xla::HloInstruction*> HasColor() {
  return ::testing::MakeMatcher(new MpmdHloInstructionMatcher(
      absl::StrCat("instruction has color"),
      [=](const HloInstruction* instruction,
          ::testing::MatchResultListener* listener) {
        return IsAssignedColor(instruction);
      }));
}

::testing::Matcher<const ::xla::HloInstruction*> TrivialShape() {
  return ::testing::MakeMatcher(new MpmdHloInstructionMatcher(
      absl::StrCat("instruction has trivial shape"),
      [=](const HloInstruction* instruction,
          ::testing::MatchResultListener* listener) {
        return ShapeUtil::ElementsIn(instruction->shape()) <= 1;
      }));
}

::testing::Matcher<const ::xla::HloInstruction*> ShapeLargerThan(
    int64_t elements) {
  return ::testing::MakeMatcher(new MpmdHloInstructionMatcher(
      absl::StrCat("instruction has shape larger than ", elements, " elements"),
      [=](const HloInstruction* instruction,
          ::testing::MatchResultListener* listener) {
        return ShapeUtil::ElementsIn(instruction->shape()) > elements;
      }));
}

::testing::Matcher<const ::xla::HloInstruction*> Color(std::string color) {
  return ::testing::MakeMatcher(new MpmdHloInstructionMatcher(
      absl::StrCat("instruction has color ", color),
      [=](const HloInstruction* instruction,
          ::testing::MatchResultListener* listener) {
        auto test_color = Color(instruction);
        if (!test_color.has_value()) {
          *listener << instruction->name() << " does not have color, expected "
                    << color;
          return false;
        }

        if (*test_color == color) {
          return true;
        }
        *listener << instruction->name() << " has color " << *test_color
                  << ", expected " << color;
        return false;
      }));
}

::testing::Matcher<const ::xla::HloInstruction*> HasLogicalAxes() {
  return ::testing::MakeMatcher(new MpmdHloInstructionMatcher(
      absl::StrCat("instruction has autosharding"),
      [=](const HloInstruction* instruction,
          ::testing::MatchResultListener* listener) {
        if (!HasAssignedAxes(instruction)) {
          *listener << instruction->name()
                    << " should not autosharding, but does not";
          return false;
        }
        return true;
      }));
}

::testing::Matcher<const SpmdHloModuleTask&> AnyTask() {
  return ::testing::MakeMatcher(new MpmdTaskMatcher(
      absl::StrCat("task can be anything"),
      [=](const SpmdHloModuleTask& task,
          ::testing::MatchResultListener* listener) { return true; }));
}

::testing::Matcher<const SpmdHloModuleTask&> LoopTask(
    std::optional<bool> has_slice) {
  return ::testing::MakeMatcher(new MpmdTaskMatcher(
      absl::StrCat("task is a loop task"),
      [=](const SpmdHloModuleTask& task,
          ::testing::MatchResultListener* listener) {
        if (!task.loop) {
          *listener << task.module->module->name()
                    << " should be a loop task, but is not";
          return false;
        }

        if (has_slice.has_value()) {
          if (*has_slice && !task.module->slice_param_number.has_value()) {
            *listener << task.module->module->name()
                      << " should have a slice parameter, but does not";
            return false;
          }
          if (!(*has_slice) && task.module->slice_param_number.has_value()) {
            *listener << task.module->module->name()
                      << " should not have a slice parameter, but does";
            return false;
          }
        }

        return true;
      }));
}

::testing::Matcher<const ::xla::Store&> Store(MpmdStoreMatcherConfig config) {
  std::string description = "store";
  if (config.type.has_value()) {
    absl::StrAppend(&description, ", type=", (int)*config.type);
  }
  if (config.index.has_value()) {
    absl::StrAppend(&description, ", index=", *config.index);
  }
  if (config.sharding.has_value()) {
    absl::StrAppend(&description, ", sharding=", config.sharding->ToString());
  }
  return ::testing::MakeMatcher(new MpmdStoreMatcher(
      std::move(description),
      [=](const ::xla::Store& store, ::testing::MatchResultListener* listener) {
        if (config.type.has_value()) {
          if (*config.type != store.type) {
            *listener << store.name << " has type " << (int)store.type
                      << ", but expected " << (int)*config.type;
            return false;
          }
        }

        if (config.index.has_value()) {
          if (*config.index != store.index) {
            *listener << store.name << " has index " << store.index
                      << ", but expected " << *config.index;
            return false;
          }
        }

        if (config.sharding.has_value()) {
          if (*config.sharding != store.mpmd_sharding) {
            *listener << store.name << " has sharding " << store.mpmd_sharding
                      << ", but expected " << *config.sharding;
            return false;
          }
        }
        return true;
      }));
}

}  // namespace mpmd_matchers

void RegisterNamedTestTask(
    std::string name, std::vector<int64_t> devices, std::vector<int64_t> dims,
    std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes) {
  RegisterMetadataNameMatcher(absl::StrCat("(", name, ")"), name);
  RegisterMetadataNameTask(name, std::move(devices), std::move(dims),
                           std::move(axes), std::move(logical_axes));
}

void RegisterMatcherTestTask(
    std::string matcher, std::vector<int64_t> devices,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes) {
  RegisterMetadataNameMatcher(matcher, std::nullopt);
  RegisterMetadataNameTask(matcher, std::move(devices), std::move(dims),
                           std::move(axes), std::move(logical_axes));
}

void RegisterMatcherTestTaskWithFactory(
    std::string matcher,
    std::function<std::vector<int64_t>(const std::string& task)> device_factory,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes) {
  RegisterMetadataNameMatcher(matcher, std::nullopt);
  RegisterMetadataNameTaskWithFactory(matcher, std::move(device_factory),
                                      std::move(dims), std::move(axes),
                                      std::move(logical_axes));
}

}  // namespace xla
