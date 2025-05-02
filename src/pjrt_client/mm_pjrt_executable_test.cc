/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mm_pjrt_executable.h"

#include <utility>

#include "gmock/gmock.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mm_pjrt_buffer.h"
#include "xla/pjrt/multimesh/mm_pjrt_client.h"
#include "xla/pjrt/multimesh/mm_sharding.h"
#include "xla/pjrt/multimesh/mm_test_base.h"
#include "xla/pjrt/multimesh/mpmd_partition.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"
#include "xla/pjrt/multimesh/zuku_execute_context.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/spmd/spmd_partitioner_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::ElementsAre;
using ::testing::Not;
using ::testing::Pointwise;
using ::testing::Property;

template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

std::vector<SpmdHloModuleTask> LoopTasks(const MultiMeshPjRtExecutable& exe) {
  std::vector<SpmdHloModuleTask> tasks;
  for (const auto& op : exe.schedule()) {
    std::visit(overloaded{[&](const SpmdHloModuleTask& task) {
                            if (task.loop) {
                              tasks.push_back(task);
                            }
                          },
                          [](const auto&) {}},
               op.op);
  }
  return tasks;
}

std::vector<SpmdHloModuleTask> NonLoopTasks(
    const MultiMeshPjRtExecutable& exe) {
  std::vector<SpmdHloModuleTask> tasks;
  for (const auto& op : exe.schedule()) {
    std::visit(overloaded{[&](const SpmdHloModuleTask& task) {
                            if (!task.loop) {
                              tasks.push_back(task);
                            }
                          },
                          [](const auto&) {}},
               op.op);
  }
  return tasks;
}

std::ostream& operator<<(std::ostream& os, const ReplicaGroup& group) {
  for (auto id : group.replica_ids()) {
    os << id << ",";
  }
  return os;
}

std::pair<std::pair<int64_t, int64_t>, std::string> TaskCallback(
    const std::string& name, bool backprop, int64_t devices_per_stage,
    int64_t layers_per_stage, int64_t total_devices) {
  int layer_num;
  bool parsed = absl::SimpleAtoi(name.substr(7), &layer_num);
  if (!parsed) {
    throw std::runtime_error(
        absl::StrCat("failed to parse layer number from", name));
  }

  std::string color = [&] {
    if (backprop) {
      return absl::StrCat("bwd.", name);
    }
    return name;
  }();

  const int64_t num_device_groups = total_devices / devices_per_stage;
  const int64_t offset =
      ((layer_num / layers_per_stage) % num_device_groups) * devices_per_stage;
  auto devices = std::make_pair(offset, offset + devices_per_stage);
  return std::make_pair(devices, std::move(color));
};

std::function<std::pair<std::pair<int64_t, int64_t>, std::string>(
    const std::string&, bool)>
TaskCallback(int64_t devices_per_stage, int64_t layers_per_stage,
             int64_t total_devices) {
  return [=](const std::string& name, bool backprop) {
    return TaskCallback(name, backprop, devices_per_stage, layers_per_stage,
                        total_devices);
  };
}

std::vector<ReplicaGroup> FlatReplicaGroups(
    const std::vector<SpmdHloModuleTask>& tasks) {
  std::vector<ReplicaGroup> all_groups;
  for (const auto& task : tasks) {
    for (auto* computation : task.compiler->optimized_module().computations()) {
      for (auto* instruction : computation->instructions()) {
        auto* casted = dynamic_cast<HloCollectiveInstruction*>(instruction);
        if (casted) {
          for (auto& group : casted->replica_groups()) {
            all_groups.push_back(group);
          }
        }
      }
    }
  }
  return all_groups;
}

bool _BufferHasSharding(PjRtBuffer* buffer, const OpSharding& sharding) {
  MultiMeshPjRtBuffer* mm_buffer = dynamic_cast<MultiMeshPjRtBuffer*>(buffer);
  if (!mm_buffer) {
    return false;
  }

  if (mm_buffer->has_sharding()) {
    HloSharding hlo_sharding = *HloSharding::FromProto(sharding);
    if (hlo_sharding != mm_buffer->sharding()) {
      std::cerr << "mismatch on " << mm_buffer->name() << ": " << hlo_sharding
                << ":" << mm_buffer->sharding() << ":"
                << mm_buffer->global_shape() << ":" << mm_buffer->name()
                << std::endl;
    }
    return mm_buffer->sharding() == hlo_sharding;
  }
  return false;
}

MATCHER_P(BufferHasSharding, sharding, "") {
  return _BufferHasSharding(arg.get(), sharding);
}

MATCHER(BufferShardingIs, "") {
  const auto& buffer = std::get<0>(arg);
  const auto& sharding = std::get<1>(arg);
  return _BufferHasSharding(buffer.get(), sharding);
}

MATCHER(GlobalShapeIs, "") {
  const auto& buffer = std::get<0>(arg);
  const auto& shape = std::get<1>(arg);
  return buffer->on_device_shape() == shape;
}

class MultiMeshExecutableTest : public MultiMeshTestBase {
 public:
  MultiMeshExecutableTest() : MultiMeshTestBase() {}

  void SetUp() override {
    MultiMeshTestBase::SetUp();
    // always enable recomputation for these tests
    EnableMultiMeshRecomputation(true);
    MultiMeshMockReset();
    SetHostOffloadMinReuseDistance(0);
    RecomputeArgumentsIfCostLessThan(0);
  }

  void TearDown() override {
    MultiMeshTestBase::TearDown();
    ReplicateParametersSmallerThanNumElements(0);
  }

  void Execute(absl::string_view hlo_module_text, int num_devices);

  void ExecutePath(absl::string_view relative_path, int num_devices);
};

void MultiMeshExecutableTest::ExecutePath(absl::string_view relative_path,
                                          int num_devices) {
  TF_ASSERT_OK_AND_ASSIGN(std::string hlo_module_text,
                          GetFileText(relative_path));
  Execute(hlo_module_text, num_devices);
}

void MultiMeshExecutableTest::Execute(absl::string_view hlo_module_text,
                                      int num_devices) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiMeshPjRtExecutable> exe,
      Compile(hlo_module_text,
              /*num_devices=*/num_devices, {.use_auto_input_sharding = true}));

  const std::vector<Shape>& parameter_shapes =
      exe->program_shape().parameters();
  auto shardings = exe->GetParameterShardings();

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_module_text));
  for (HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter &&
        instruction->has_sharding()) {
      TF_ASSERT_OK_AND_ASSIGN(HloSharding test_sharding,
                              HloSharding::FromProto(shardings->at(
                                  instruction->parameter_number())));
      ASSERT_EQ(instruction->sharding(), test_sharding);
    }
  }

  ASSERT_TRUE(shardings.has_value());
  ASSERT_EQ(parameter_shapes.size(), shardings->size());

  size_t num_parameters = parameter_shapes.size();
  std::vector<std::unique_ptr<MultiMeshPjRtBuffer>> parameters;
  parameters.reserve(num_parameters);
  for (size_t parameter_number = 0; parameter_number < parameter_shapes.size();
       ++parameter_number) {
    TF_ASSERT_OK_AND_ASSIGN(
        HloSharding param_sharding,
        HloSharding::FromProto(shardings->at(parameter_number)));
    Shape global_shape = parameter_shapes[parameter_number];
    Shape shard_shape = spmd::MakePartitionedShape(
        parameter_shapes[parameter_number], param_sharding);
    std::string name = absl::StrCat("parameter_", parameter_number);
    const int sharding_num_devices =
        param_sharding.IsReplicated()
            ? num_devices
            : param_sharding.tile_assignment().num_elements();

    const int first_device = param_sharding.IsReplicated()
                                 ? 0
                                 : param_sharding.tile_assignment().first();

    zuku::DeviceList dl =
        zuku::DeviceList::Create(first_device, sharding_num_devices);
    TF_ASSERT_OK_AND_ASSIGN(
        zuku::ShardedShape sharded_shape,
        XlaShapeToZukuShape(global_shape, dl, param_sharding));
    StoreHandle store = client_->mutable_context()->CreateStore(
        /*local_device_id=*/0, /*global_device_id=*/0, std::move(sharded_shape),
        {.name = name, .allocate_from_cache = false});
    auto mm_buffer = std::make_unique<MultiMeshPjRtBuffer>(
        store, std::move(param_sharding), std::move(global_shape),
        std::move(shard_shape), exe->mm_client(), exe->base_client(),
        exe->addressable_devices()[0],
        exe->addressable_devices()[0]->default_memory_space().value_or(nullptr),
        std::move(name));
    parameters.push_back(std::move(mm_buffer));
  }

  std::vector<PjRtBuffer*> param_buffer_pointers;
  param_buffer_pointers.reserve(parameters.size());
  for (auto&& param : parameters) {
    param_buffer_pointers.push_back(param.get());
  }

  ExecuteOptions options;
  options.arguments_are_tupled = false;

  std::optional<std::vector<PjRtFuture<>>> returned_futures;
  TF_ASSERT_OK_AND_ASSIGN(
      auto outputs,
      exe->Execute({param_buffer_pointers}, options, returned_futures));

  auto output_shardings = exe->GetOutputShardings();
  ASSERT_TRUE(output_shardings.has_value());
  EXPECT_THAT(outputs[0], Pointwise(BufferShardingIs(), *output_shardings));
  TF_ASSERT_OK_AND_ASSIGN(auto output_shapes, exe->GetOutputShapes());

  if (output_shapes[0].IsTuple()) {
    EXPECT_THAT(outputs[0],
                Pointwise(GlobalShapeIs(), output_shapes[0].tuple_shapes()));
  } else {
    EXPECT_THAT(outputs[0], Pointwise(GlobalShapeIs(), {output_shapes[0]}));
  }

  module->input_output_alias_config().ForEachAlias(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias& alias) {
        int64_t output = output_index.empty() ? 0 : output_index.front();
        EXPECT_THAT(outputs[0][output],
                    BufferHasSharding((*shardings)[alias.parameter_number]));
      });
}

static constexpr absl::string_view kShardMatchInputOutputAlias = R"(
HloModule jit_f, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }
ENTRY %main.12 {
  Arg_0.1 = f32[4,4]{1,0} parameter(0)
  custom-call.7 = f32[4,4]{1,0} custom-call(Arg_0.1), custom_call_target="Sharding", sharding={replicated}
  Arg_2.3 = f32[4,4]{1,0} parameter(1), sharding={devices=[2,1]0,1}
  add.1 = f32[4,4]{1,0} add(custom-call.7, Arg_2.3)
  add.3 = f32[4,4]{1,0} add(add.1, Arg_2.3)
  ROOT tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.1, add.3)
}
)";

TEST_F(MultiMeshExecutableTest, ShardMatchInputOutputAlias) {
  GTEST_SKIP() << "unsupported shape";
  Execute(kShardMatchInputOutputAlias, /*num_devices=*/2);
}

static constexpr absl::string_view kMultipleMicrobatchSlicesHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4]{0}, f32[4]{0})->f32[]}, allow_spmd_sharding_propagation_to_parameters={true,true,true,true}, allow_spmd_sharding_propagation_to_output={true}

f.impl.11 {
  Arg_0.12 = f32[2,4]{1,0} parameter(0)
  Arg_1.13 = f32[4]{0} parameter(1)
  reshape.14 = f32[1,4]{1,0} reshape(Arg_1.13)
  broadcast.15 = f32[1,4]{1,0} broadcast(reshape.14), dimensions={0,1}
  reshape.16 = f32[4]{0} reshape(broadcast.15)
  broadcast.17 = f32[2,4]{1,0} broadcast(reshape.16), dimensions={1}
  ROOT multiply.18 = f32[2,4]{1,0} multiply(Arg_0.12, broadcast.17)
} // f.impl.11

f.impl.19 {
  Arg_0.20 = f32[2,4]{1,0} parameter(0)
  Arg_1.21 = f32[4]{0} parameter(1)
  reshape.22 = f32[1,4]{1,0} reshape(Arg_1.21)
  broadcast.23 = f32[1,4]{1,0} broadcast(reshape.22), dimensions={0,1}
  reshape.24 = f32[4]{0} reshape(broadcast.23)
  broadcast.25 = f32[2,4]{1,0} broadcast(reshape.24), dimensions={1}
  ROOT multiply.26 = f32[2,4]{1,0} multiply(Arg_0.20, broadcast.25)
} // f.impl.19

region_1.27 {
  Arg_0.28 = f32[] parameter(0)
  Arg_1.29 = f32[] parameter(1)
  ROOT add.30 = f32[] add(Arg_0.28, Arg_1.29)
}

g.impl.31 {
  Arg_0.32 = f32[2,4]{1,0} parameter(0)
  Arg_1.33 = f32[4]{0} parameter(1)
  reshape.35 = f32[1,4]{1,0} reshape(Arg_1.33)
  broadcast.36 = f32[1,4]{1,0} broadcast(reshape.35), dimensions={0,1}
  reshape.37 = f32[4]{0} reshape(broadcast.36)
  broadcast.38 = f32[2,4]{1,0} broadcast(reshape.37), dimensions={1}
  multiply.39 = f32[2,4]{1,0} multiply(Arg_0.32, broadcast.38)
  constant.34 = f32[] constant(0)
  ROOT reduce.40 = f32[] reduce(multiply.39, constant.34), dimensions={0,1}, to_apply=region_1.27
} // g.impl.31

region_1.41 {
  Arg_0.42 = f32[] parameter(0)
  Arg_1.43 = f32[] parameter(1)
  ROOT add.44 = f32[] add(Arg_0.42, Arg_1.43)
}

g.impl.45 {
  Arg_0.46 = f32[2,4]{1,0} parameter(0)
  Arg_1.47 = f32[4]{0} parameter(1)
  reshape.49 = f32[1,4]{1,0} reshape(Arg_1.47)
  broadcast.50 = f32[1,4]{1,0} broadcast(reshape.49), dimensions={0,1}
  reshape.51 = f32[4]{0} reshape(broadcast.50)
  broadcast.52 = f32[2,4]{1,0} broadcast(reshape.51), dimensions={1}
  multiply.53 = f32[2,4]{1,0} multiply(Arg_0.46, broadcast.52)
  constant.48 = f32[] constant(0)
  ROOT reduce.54 = f32[] reduce(multiply.53, constant.48), dimensions={0,1}, to_apply=region_1.41
} // g.impl.45

None.55 {
  Arg_0.56 = f32[4,4]{1,0} parameter(0)
  Arg_4.60 = s32[] parameter(4)
  constant.64 = s32[] constant(0)
  compare.65 = pred[] compare(Arg_4.60, constant.64), direction=LT
  constant.63 = s32[] constant(4)
  add.66 = s32[] add(Arg_4.60, constant.63)
  select.67 = s32[] select(compare.65, add.66, Arg_4.60)
  dynamic-slice.68 = f32[2,4]{1,0} dynamic-slice(Arg_0.56, select.67, constant.64), dynamic_slice_sizes={2,4}
  custom-call.69 = f32[2,4]{1,0} custom-call(dynamic-slice.68), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  Arg_1.57 = f32[4,4]{1,0} parameter(1)
  compare.70 = pred[] compare(Arg_4.60, constant.64), direction=LT
  add.71 = s32[] add(Arg_4.60, constant.63)
  select.72 = s32[] select(compare.70, add.71, Arg_4.60)
  dynamic-slice.73 = f32[2,4]{1,0} dynamic-slice(Arg_1.57, select.72, constant.64), dynamic_slice_sizes={2,4}
  custom-call.74 = f32[2,4]{1,0} custom-call(dynamic-slice.73), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  multiply.76 = f32[2,4]{1,0} multiply(custom-call.69, custom-call.74)
  Arg_2.58 = f32[4]{0} parameter(2)
  call.77 = f32[2,4]{1,0} call(multiply.76, Arg_2.58), to_apply=f.impl.11
  custom-call.78 = f32[2,4]{1,0} custom-call(multiply.76, Arg_2.58), custom_call_target="MultiMeshTask", called_computations={f.impl.19}, backend_config={"name": "f", "devices": [0], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  Arg_3.59 = f32[4]{0} parameter(3)
  call.79 = f32[] call(custom-call.78, Arg_3.59), to_apply=g.impl.31
  constant.62 = s32[] constant(2)
  add.75 = s32[] add(Arg_4.60, constant.62)
  Arg_5.61 = f32[] parameter(5)
  custom-call.80 = f32[] custom-call(custom-call.78, Arg_3.59), custom_call_target="MultiMeshTask", called_computations={g.impl.45}, backend_config={"name": "g", "devices": [0], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  add.81 = f32[] add(Arg_5.61, custom-call.80)
  ROOT tuple.82 = (s32[], f32[]) tuple(add.75, add.81)
} // None.55

region_0.83 {
  arg_tuple.84 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) parameter(0)
  get-tuple-element.85 = s32[] get-tuple-element(arg_tuple.84), index=0
  constant.92 = s32[] constant(1)
  add.96 = s32[] add(get-tuple-element.85, constant.92)
  get-tuple-element.88 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=3
  get-tuple-element.89 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=4
  get-tuple-element.90 = f32[4]{0} get-tuple-element(arg_tuple.84), index=5
  get-tuple-element.91 = f32[4]{0} get-tuple-element(arg_tuple.84), index=6
  get-tuple-element.86 = s32[] get-tuple-element(arg_tuple.84), index=1
  get-tuple-element.87 = f32[] get-tuple-element(arg_tuple.84), index=2
  call.93 = (s32[], f32[]) call(get-tuple-element.88, get-tuple-element.89, get-tuple-element.90, get-tuple-element.91, get-tuple-element.86, /*index=5*/get-tuple-element.87), to_apply=None.55
  get-tuple-element.94 = s32[] get-tuple-element(call.93), index=0
  get-tuple-element.95 = f32[] get-tuple-element(call.93), index=1
  ROOT tuple.97 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) tuple(add.96, get-tuple-element.94, get-tuple-element.95, get-tuple-element.88, get-tuple-element.89, /*index=5*/get-tuple-element.90, get-tuple-element.91)
} // region_0.83

region_2.98 {
  arg_tuple.99 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) parameter(0)
  get-tuple-element.101 = s32[] get-tuple-element(arg_tuple.99), index=1
  get-tuple-element.102 = f32[] get-tuple-element(arg_tuple.99), index=2
  get-tuple-element.103 = f32[4,4]{1,0} get-tuple-element(arg_tuple.99), index=3
  get-tuple-element.104 = f32[4,4]{1,0} get-tuple-element(arg_tuple.99), index=4
  get-tuple-element.105 = f32[4]{0} get-tuple-element(arg_tuple.99), index=5
  get-tuple-element.106 = f32[4]{0} get-tuple-element(arg_tuple.99), index=6
  get-tuple-element.100 = s32[] get-tuple-element(arg_tuple.99), index=0
  constant.107 = s32[] constant(2)
  ROOT compare.108 = pred[] compare(get-tuple-element.100, constant.107), direction=LT
} // region_2.98

ENTRY main.117 {
  constant.5 = s32[] constant(0)
  constant.6 = f32[] constant(0)
  custom-call.7 = f32[] custom-call(constant.6), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  Arg_0.1 = f32[4,4]{1,0} parameter(0)
  custom-call.8 = f32[4,4]{1,0} custom-call(Arg_0.1), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  Arg_1.2 = f32[4,4]{1,0} parameter(1)
  custom-call.9 = f32[4,4]{1,0} custom-call(Arg_1.2), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  Arg_2.3 = f32[4]{0} parameter(2)
  Arg_3.4 = f32[4]{0} parameter(3)
  tuple.10 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) tuple(constant.5, constant.5, custom-call.7, custom-call.8, custom-call.9, /*index=5*/Arg_2.3, Arg_3.4)
  while.109 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) while(tuple.10), condition=region_2.98, body=region_0.83
  get-tuple-element.110 = s32[] get-tuple-element(while.109), index=0
  get-tuple-element.111 = s32[] get-tuple-element(while.109), index=1
  ROOT get-tuple-element.112 = f32[] get-tuple-element(while.109), index=2
  get-tuple-element.113 = f32[4,4]{1,0} get-tuple-element(while.109), index=3
  get-tuple-element.114 = f32[4,4]{1,0} get-tuple-element(while.109), index=4
  get-tuple-element.115 = f32[4]{0} get-tuple-element(while.109), index=5
  get-tuple-element.116 = f32[4]{0} get-tuple-element(while.109), index=6
} // main.117
)";

TEST_F(MultiMeshExecutableTest, ExecuteMicrobatches) {
  Execute(kMultipleMicrobatchSlicesHlo, /*num_devices=*/2);
}

TEST_F(MultiMeshExecutableTest, Pipeline2x8Stages) {
  GTEST_SKIP() << "Data parallelism with non-uniform mesh sizes has sharding "
                  "propagation bug with aliasing";

  static constexpr int kLayersPerStage = 12;
  static constexpr int kDevicesPerStage = 8;
  static constexpr int kTotalDevices = 16;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {4, 1, 2},
      {"x", "y", "z"}, transformer_axes);
  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "(final_ln).*",
                         "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices}, {2, 1, 8},
                            {"x", "y", "z"}, embeddings_axes);
  }

  ExecutePath("pipeline_16gpus.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, Pipeline2x8StagesReplicateSmallParams) {
  static constexpr int kLayersPerStage = 1;
  static constexpr int kDevicesPerStage = 8;
  static constexpr int kTotalDevices = 16;
  static constexpr int kNumDeviceGroups = 2;
  static constexpr int64_t kMaxBytesAllocated = 29500000000;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {1, 1, 8},
      {"x", "y", "z"}, transformer_axes);
  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "(default)"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage}, {1, 1, 8},
                            {"x", "y", "z"}, embeddings_axes);
  }

  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher,
                            {kTotalDevices - kDevicesPerStage, kTotalDevices},
                            {1, 1, 8}, {"x", "y", "z"}, embeddings_axes);
  }

  ReplicateParametersSmallerThanNumElements(1024 * 2048);

  ExecutePath("pipeline_16gpus.txt", /*num_devices=*/kTotalDevices);

  EXPECT_LT(DeviceBytesHighWatermark(0), kMaxBytesAllocated);
}

TEST_F(MultiMeshExecutableTest, Pipeline2x8Stages24LayersReplicateSmallParams) {
  static constexpr int kLayersPerStage = 2;
  static constexpr int kTotalDevices = 16;
  static constexpr int kDevicesPerStage = 8;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"},
      {"data", "y"},
      {"mdl", "z"},
      {"seq", "y"},
      {"seq", "z"}};

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {1, 1, 8},
      {"x", "y", "z"}, transformer_axes);
  for (auto&& matcher : {"(emb_lookup).*", "(position_emb).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage}, {1, 1, 8},
                            {"x", "y", "z"}, embeddings_axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher,
                            {kTotalDevices - kDevicesPerStage, kTotalDevices},
                            {1, 1, 8}, {"x", "y", "z"}, embeddings_axes);
  }

  ReplicateParametersSmallerThanNumElements(64 * 2048);

  ExecutePath("pipeline_24layers_16gpus.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, Opt175ReplicateSmallParams) {
  static constexpr int kLayersPerStage = 1;
  static constexpr int kTotalDevices = 128;
  static constexpr int kDevicesPerStage = 8;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"},
      {"data", "y"},
      {"mdl", "z"},
      {"seq", "y"},
      {"seq", "z"}};

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {1, 1, 8},
      {"x", "y", "z"}, transformer_axes);
  for (auto&& matcher : {"(emb_lookup).*", "(position_emb).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage}, {1, 1, 8},
                            {"x", "y", "z"}, embeddings_axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher,
                            {kTotalDevices - kDevicesPerStage, kTotalDevices},
                            {1, 1, 8}, {"x", "y", "z"}, embeddings_axes);
  }

  ReplicateParametersSmallerThanNumElements(128 * 2048);

  ExecutePath("opt175_replicated_params.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, Pipeline2x8StagesMicrobatch8) {
  GTEST_SKIP() << "TODO: sharding propagation bug leads to bad alias";

  static constexpr int kTotalDevices = 16;
  static constexpr int kDevicesPerStage = 8;
  static constexpr int kLayersPerStage = 12;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {4, 1, 2},
      {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "(final_ln).*",
                         "(compute_loss).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage}, {4, 1, 2},
                            {"x", "y", "z"}, embeddings_axes);
  }

  ExecutePath("2x8_mb8_distributed_embeddings.txt",
              /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, Pipeline2x8StagesCircularScheduling) {
  GTEST_SKIP() << "Need to resolve sharding mismatch with data parallelism";

  static constexpr int kTotalDevices = 16;
  static constexpr int kLayersPerStage = 4;
  static constexpr int kDevicesPerStage = 8;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };
  std::vector<int64_t> embeddings_dims = {2, 1, 8};

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {4, 1, 2},
      {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "(final_ln).*",
                         "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices}, {2, 1, 8},
                            {"x", "y", "z"}, embeddings_axes);
  }

  ExecutePath("pipeline_16gpus.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest,
       Pipeline2x8StagesCircularSchedulingControlReplication) {
  GTEST_SKIP() << "Need to resolve sharding mismatch with data parallelism";

  static constexpr int kTotalDevices = 16;
  static constexpr int kLayersPerStage = 4;
  static constexpr int kDevicesPerStage = 8;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };
  std::vector<int64_t> embeddings_dims = {2, 1, 8};

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {4, 1, 2},
      {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "(final_ln).*",
                         "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices}, {2, 1, 8},
                            {"x", "y", "z"}, embeddings_axes);
  }

  static constexpr int kNumRepeats = 4;
  MultiMeshMockSetComputeHashes(true);
  ExecutePath("pipeline_16gpus.txt", /*num_devices=*/kTotalDevices);
  int64_t first_hash = MultiMeshMockStateHash();
  VLOG(3) << "test=" << first_hash;
  for (int repeat = 0; repeat < kNumRepeats; ++repeat) {
    MultiMeshMockReset();
    MultiMeshMockSetComputeHashes(true);
    ExecutePath("pipeline_16gpus.txt", /*num_devices=*/kTotalDevices);
    int64_t test_hash = MultiMeshMockStateHash();
    VLOG(3) << "test=" << test_hash;
    ASSERT_EQ(first_hash, test_hash);
  }
}

TEST_F(MultiMeshExecutableTest, Pipeline4x2Stages) {
  GTEST_SKIP() << "TODO: fix non-uniform sharding propagation bug";
  static constexpr int kDevicesPerStage = 2;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kTotalDevices = 8;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };
  std::vector<int64_t> embeddings_dims = {2, 1, 4};

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {2, 1, 1},
      {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "(final_ln).*",
                         "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices}, {2, 1, 4},
                            {"x", "y", "z"}, embeddings_axes);
  }

  ExecutePath("4stage_pipeline.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, Pipeline4x2Stages1F1B) {
  GTEST_SKIP() << "Sharding pattern not yet supported";

  static constexpr int kDevicesPerStage = 8;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kTotalDevices = 32;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {1, 1, 8},
      {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage}, {1, 1, 8},
                            {"x", "y", "z"}, embeddings_axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher,
                            {kTotalDevices - kDevicesPerStage, kTotalDevices},
                            {1, 1, 8}, {"x", "y", "z"}, embeddings_axes);
  }

  ReplicateParametersSmallerThanNumElements(64 * 2048);
  ExecutePath("4stage_1f1b_pipeline.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, MaximalDeviceSharding4x1Stages) {
  GTEST_SKIP() << "do not yet support maximal device sharding";

  static constexpr int kDevicesPerStage = 1;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kTotalDevices = 4;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {1, 1, 1},
      {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, 1}, {1, 1, 1}, {"x", "y", "z"},
                            embeddings_axes);
  }

  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher, {3, 4}, {1, 1, 1}, {"x", "y", "z"},
                            embeddings_axes);
  }

  ExecutePath("maximal_sharding_4stages.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, ShardedPipeline4x2Stages) {
  GTEST_SKIP() << "TODO: fix non-uniform sharding propagation bug";
  static constexpr int kTotalDevices = 8;
  static constexpr int kDevicesPerStage = 2;
  static constexpr int kLayersPerStage = 1;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };
  std::vector<int64_t> embeddings_dims = {2, 1, 4};

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {2, 1, 1},
      {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "(final_ln).*",
                         "(compute_loss).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices}, {2, 1, 4},
                            {"x", "y", "z"}, embeddings_axes);
  }

  ExecutePath("sharded_args_4stages.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, DP2_PP2_TP8_4nodes) {
  static constexpr int kDevicesPerStage = 16;
  static constexpr int kLayersPerStage = 12;
  static constexpr int kEmbeddingsDevices = 16;
  static constexpr int kLogitsDevices = 16;
  static constexpr int kTotalDevices = kEmbeddingsDevices + kLogitsDevices;

  std::vector<int64_t> embeddings_dims = {2, 1, 8};

  std::vector<int64_t> logits_dims = {2, 1, 8};

  std::vector<std::pair<std::string, std::string>> axes = {
      {"replica", "x"},
      {"data", "y"},
      {"mdl", "z"},
      // for small tensors, the very last thing to shard is along
      // the sequence dimension
      {"seq", "z"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {2, 1, 8},
      {"x", "y", "z"}, axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kEmbeddingsDevices}, {2, 1, 8},
                            {"x", "y", "z"}, axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher,
                            {kTotalDevices - kLogitsDevices, kTotalDevices},
                            {2, 1, 8}, {"x", "y", "z"}, axes);
  }

  ExecutePath("data-parallel-4-nodes.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, ShardedPipeline2x4StagesSmallMicrobatch) {
  static constexpr int kTotalDevices = 8;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kDevicesPerStage = 4;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };
  std::vector<int64_t> embeddings_dims = {2, 1, 4};

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {2, 1, 2},
      {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "(final_ln).*",
                         "(compute_loss).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices}, {2, 1, 4},
                            {"x", "y", "z"}, embeddings_axes);
  }

  ExecutePath("2x4pipeline_mb2.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, PP2_TP4_TransformerEngine) {
  static constexpr int kTotalDevices = 8;
  static constexpr int kDevicesPerStage = 4;
  static constexpr int kLayersPerStage = 1;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices), {1, 1, 4},
      {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage}, {1, 1, 4},
                            {"x", "y", "z"}, embeddings_axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher,
                            {kTotalDevices - kDevicesPerStage, kTotalDevices},
                            {1, 1, 4}, {"x", "y", "z"}, embeddings_axes);
  }

  ExecutePath("tp4_pp2_te.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, TransformerTwoNodes) {
  static constexpr int kTotalDevices = 16;
  static constexpr int kDevicesPerStage = 4;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kNumGroups = 4;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices),
      {1, 1, kDevicesPerStage}, {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"},
                            embeddings_axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(
        matcher, {kTotalDevices - kDevicesPerStage, kTotalDevices},
        {1, 1, kDevicesPerStage}, {"x", "y", "z"}, embeddings_axes);
  }

  ReplicateParametersSmallerThanNumElements(16 * 2048);
  ExecutePath("2nodes_transformer.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, TransformerTwoNodesHostOffload) {
  GTEST_SKIP() << "host offloading not yet supported";
  static constexpr int kTotalDevices = 16;
  static constexpr int kDevicesPerStage = 4;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kNumGroups = 4;
  static constexpr int64_t kMaxDeviceBytesHighWatermark = 53 * 1e9;
  static constexpr int64_t kMaxHostBytesHighWatermark = 50 * 1e9;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices),
      {1, 1, kDevicesPerStage}, {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"},
                            embeddings_axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(
        matcher, {kTotalDevices - kDevicesPerStage, kTotalDevices},
        {1, 1, kDevicesPerStage}, {"x", "y", "z"}, embeddings_axes);
  }

  ReplicateParametersSmallerThanNumElements(16 * 2048);
  RecomputeArgumentsIfCostLessThan(1024 * 1024);
  ExecutePath("2nodes_transformer.txt", /*num_devices=*/kTotalDevices);

  // without offloading, the device byte limit should be exceeded
  // make sure this is verified, the test should fail if host offloading fails
  EXPECT_GT(DeviceBytesHighWatermark(/*local_device_id=*/0),
            kMaxDeviceBytesHighWatermark);
  EXPECT_EQ(HostBytesHighWatermark(/*local_device_id=*/0), 0);
  // Any intermediates that are not immediately reused, offload to host
  MultiMeshMockReset();
  SetHostOffloadMinReuseDistance(2);
  SetHostOffloadMinSize(100e3);

  ExecutePath("2nodes_transformer.txt", /*num_devices=*/kTotalDevices);

  EXPECT_LT(DeviceBytesHighWatermark(/*local_device_id=*/0),
            kMaxDeviceBytesHighWatermark);
  EXPECT_LT(HostBytesHighWatermark(/*local_device_id=*/0),
            kMaxHostBytesHighWatermark);
}

TEST_F(MultiMeshExecutableTest, ShardingMismatchLoadBalanced) {
  GTEST_SKIP() << "Load balancing not yet implemented";
  static constexpr int kTotalDevices = 8;
  static constexpr int kDevicesPerStage = 4;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kNumGroups = 2;

  std::vector<std::pair<std::string, std::string>> axes = {
      {"data", "x"},     {"stage", "y"},  {"fsdp", "y"},
      {"sequence", "z"}, {"tensor", "z"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices),
      {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  for (auto&& matcher :
       {"(position_emb).*", "(emb_lookup).*", "(token_embedder).*"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  }
  for (auto&& matcher :
       {"(final_ln).*", "(decoder_norm).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  }

  ReplicateParametersSmallerThanNumElements(128 * 2048);
  ExecutePath("sharding_mismatch_load_balanced_2nodes.txt",
              /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, HostOffloadCrash) {
  GTEST_SKIP() << "Host offloading support not yet available";
  static constexpr int kTotalDevices = 8;
  static constexpr int kDevicesPerStage = 4;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kNumGroups = 2;

  std::vector<std::pair<std::string, std::string>> axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices),
      {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  }

  ReplicateParametersSmallerThanNumElements(128 * 2048);
  ExecutePath("host_offload_crash_2layers.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest,
       TransformerTwoNodesDynamicSliceEmbeddingsLogits) {
  GTEST_SKIP() << "Load balancing not yet implemented";

  static constexpr int kTotalDevices = 16;
  static constexpr int kDevicesPerStage = 4;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kNumGroups = 4;

  std::vector<std::pair<std::string, std::string>> embeddings_axes = {
      {"replica", "x"}, {"seq", "y"},     {"seq", "z"},
      {"mdl", "z"},     {"replica", "y"}, {"replica", "z"},
  };

  std::vector<std::pair<std::string, std::string>> transformer_axes = {
      {"replica", "x"}, {"data", "y"}, {"mdl", "z"},
      {"seq", "y"},     {"seq", "z"},  {"mdl", "x"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices),
      {1, 1, kDevicesPerStage}, {"x", "y", "z"}, transformer_axes);

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"},
                            embeddings_axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"},
                            embeddings_axes);
  }

  ReplicateParametersSmallerThanNumElements(16 * 2048);
  ExecutePath("2nodes_transformer.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, Gpt3_16nodes) {
  GTEST_SKIP() << "Load balancing not yet implemented";
  static constexpr int kTotalDevices = 128;
  static constexpr int kDevicesPerStage = 8;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kNumGroups = 16;

  std::vector<std::pair<std::string, std::string>> axes = {
      {"data", "x"},           {"stage", "y"},    {"fsdp", "y"},
      {"fsdp_transpose", "y"}, {"sequence", "z"}, {"tensor", "z"},
      {"autoregressive", "z"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices),
      {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  for (auto&& matcher :
       {"(position_emb).*", "(emb_lookup).*", "(token_embedder).*"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  }
  for (auto&& matcher : {"(final_ln).*", "(compute_loss).*", "(logits_dense).*",
                         "(decoder_norm).*"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  }

  ReplicateParametersSmallerThanNumElements(256 * 2048);
  ExecutePath("maxtext_gpt3.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, MaxtextMismatchedAlias) {
  static constexpr int kTotalDevices = 8;
  static constexpr int kDevicesPerStage = 4;
  static constexpr int kLayersPerStage = 1;
  static constexpr int kNumGroups = 2;

  std::vector<std::pair<std::string, std::string>> axes = {
      {"data", "x"},           {"stage", "y"},    {"fsdp", "y"},
      {"fsdp_transpose", "y"}, {"sequence", "z"}, {"tensor", "z"},
      {"autoregressive", "z"},
  };

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices),
      {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  for (auto&& matcher : {"(emb).*", "(final_ln).*", "(compute_loss).*",
                         "(logits_dense).*", "(decoder_norm).*", "default"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage},
                            {1, 1, kDevicesPerStage}, {"x", "y", "z"}, axes);
  }

  ReplicateParametersSmallerThanNumElements(8 * 128);
  ExecutePath("maxtext_mismatched_alias.txt", /*num_devices=*/kTotalDevices);
}

TEST_F(MultiMeshExecutableTest, DP_2x2x2) {
  std::vector<std::pair<std::string, std::string>> axes = {
      {"replica", "x"},
      {"data", "y"},
      {"mdl", "z"},
  };

  constexpr int kTotalDevices = 8;
  constexpr int kLayersPerStage = 1;
  constexpr int kNumGroups = 2;
  constexpr int kDevicesPerStage = 4;
  constexpr int kDP = 2;
  constexpr int kTP = 2;

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices),
      {kDP, 1, kTP}, {"x", "y", "z"}, axes);

  for (auto&& matcher : {"(emb).*", "(final_ln).*", "(compute_loss).*",
                         "(logits_dense).*", "(decoder_norm).*"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage}, {kDP, 1, kTP},
                            {"x", "y", "z"}, axes);
  }

  SetCompileModules(true);
  ReplicateParametersSmallerThanNumElements(256 * 2048);

  TF_ASSERT_OK_AND_ASSIGN(std::string hlo_module_text,
                          GetFileText("paxml_dp2_pp2_tp2.txt"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiMeshPjRtExecutable> exe,
      Compile(hlo_module_text,
              /*num_devices=*/kTotalDevices,
              {.use_auto_input_sharding = true, .hoist_loop_convert = true}));

  auto loop_replica_groups = FlatReplicaGroups(LoopTasks(*exe));
  EXPECT_THAT(loop_replica_groups,
              Not(Contains(Property(
                  &ReplicaGroup::replica_ids,
                  AnyOf(ElementsAre(0, 2), ElementsAre(0, 1, 2, 3))))));

  auto post_loop_replica_groups = FlatReplicaGroups(NonLoopTasks(*exe));
  EXPECT_THAT(
      post_loop_replica_groups,
      Contains(Property(&ReplicaGroup::replica_ids,
                        AnyOf(ElementsAre(0, 2), ElementsAre(0, 1, 2, 3)))));

  // strip out reduces instead of executing them
  {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<MultiMeshPjRtExecutable> exe,
                            Compile(hlo_module_text,
                                    /*num_devices=*/kTotalDevices,
                                    {.use_auto_input_sharding = true,
                                     .hoist_loop_convert = true,
                                     .remove_hoisted_reduces = true}));

    // there should be no DP collectives inside our outside the loop

    auto loop_replica_groups = FlatReplicaGroups(LoopTasks(*exe));
    EXPECT_THAT(loop_replica_groups,
                Not(Contains(Property(
                    &ReplicaGroup::replica_ids,
                    AnyOf(ElementsAre(0, 2), ElementsAre(0, 1, 2, 3))))));

    auto post_loop_replica_groups = FlatReplicaGroups(NonLoopTasks(*exe));
    EXPECT_THAT(post_loop_replica_groups,
                Not(Contains(Property(
                    &ReplicaGroup::replica_ids,
                    AnyOf(ElementsAre(0, 2), ElementsAre(0, 1, 2, 3))))));
  }
}

TEST_F(MultiMeshExecutableTest, DP_2_2_8_unused_loop_outputs) {
  std::vector<std::pair<std::string, std::string>> axes = {
      {"data", "x"},
      {"tensor", "z"},
  };

  constexpr int kTotalDevices = 8;
  constexpr int kLayersPerStage = 1;
  constexpr int kNumGroups = 2;
  constexpr int kDevicesPerStage = 4;
  constexpr int kDP = 2;
  constexpr int kTP = 2;

  RegisterMatcherTestTaskWithFactory(
      "(layers_\\d+)",
      TaskCallback(kDevicesPerStage, kLayersPerStage, kTotalDevices),
      {kDP, 1, kTP}, {"x", "y", "z"}, axes);

  for (auto&& matcher : {"(emb).*", "(final_ln).*", "(compute_loss).*",
                         "(logits_dense).*", "(decoder_norm).*"}) {
    RegisterMatcherTestTask(matcher, {0, kDevicesPerStage}, {kDP, 1, kTP},
                            {"x", "y", "z"}, axes);
  }

  SetCompileModules(true);
  ReplicateParametersSmallerThanNumElements(256 * 2048);

  TF_ASSERT_OK_AND_ASSIGN(std::string hlo_module_text,
                          GetFileText("unused_loop_output_dp_transformer.txt"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MultiMeshPjRtExecutable> exe,
      Compile(hlo_module_text,
              /*num_devices=*/kTotalDevices,
              {.use_auto_input_sharding = true, .hoist_loop_convert = true}));

  auto loop_replica_groups = FlatReplicaGroups(LoopTasks(*exe));
  EXPECT_THAT(loop_replica_groups,
              Not(Contains(Property(
                  &ReplicaGroup::replica_ids,
                  AnyOf(ElementsAre(0, 2), ElementsAre(0, 1, 2, 3))))));

  auto post_loop_replica_groups = FlatReplicaGroups(NonLoopTasks(*exe));
  EXPECT_THAT(
      post_loop_replica_groups,
      Contains(Property(&ReplicaGroup::replica_ids,
                        AnyOf(ElementsAre(0, 2), ElementsAre(0, 1, 2, 3)))));
}

TEST_F(MultiMeshExecutableTest, SingleNodeTransformer) {
  GTEST_SKIP() << "TODO: sharding propagation bug";
  static constexpr int kTotalDevices = 8;

  std::vector<int64_t> dims = {1, 8, 1};

  std::vector<std::pair<std::string, std::string>> axes = {{"replica", "x"},
                                                           {"data", "y"},
                                                           {"mdl", "z"},
                                                           {"seq", "y"},
                                                           {"seq", "z"}};

  for (auto&& matcher : {"(position_emb).*", "(emb_lookup).*", "(final_ln).*",
                         "(compute_loss).*", "default", "(layers_\\d+)"}) {
    RegisterMatcherTestTask(matcher, {0, kTotalDevices}, dims, {"x", "y", "z"},
                            axes);
  }
  ExecutePath("single-node-opt.txt", /*num_devices=*/kTotalDevices);
}

}  // namespace
}  // namespace xla
