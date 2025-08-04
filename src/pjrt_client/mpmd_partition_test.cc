/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "xla/pjrt/multimesh/mpmd_partition.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/client/executable_build_options.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::IsEmpty;
using ::testing::IsSupersetOf;
using ::testing::Not;
using ::testing::Pointwise;
using ::testing::UnorderedElementsAre;

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

MATCHER_P(NumOutputs, num, "") {
  const SpmdHloModuleTask& task = arg;
  return task.outputs.size() == num;
}

MATCHER(IsLoopTask, "") {
  const auto& task = std::get<0>(arg);
  bool should_be_loop = std::get<1>(arg);
  bool has_loop_config = task.loop_config != nullptr;
  return has_loop_config == should_be_loop;
}

MATCHER(TrivialStoreShape, "") {
  auto&& store = arg;
  return ShapeUtil::ElementsIn(store.shape) == 1;
}

MATCHER(TrivialStoreSharding, "") {
  auto&& store = arg;
  return store.mpmd_sharding.IsReplicated();
}

MATCHER(NonTrivialStoreShape, "") {
  auto&& store = arg;
  return ShapeUtil::ElementsIn(store.shape) > 1;
}

MATCHER(NonTrivialStoreSharding, "") {
  auto&& store = arg;
  return !store.sharding.IsReplicated();
}

std::vector<std::vector<std::pair<Store, Store>>> GetAliasPairs(
    const std::vector<SpmdHloModuleTask>& tasks) {
  std::vector<std::vector<std::pair<Store, Store>>> stores;
  for (auto& task : tasks) {
    std::vector<std::pair<Store, Store>> pair_vec;
    task.module->module->input_output_alias_config().ForEachAlias(
        [&](const ShapeIndex& output_index,
            const HloInputOutputAliasConfig::Alias& alias) {
          const int output_number = [&] {
            if (output_index.empty()) {
              return 0;
            }
            return (int)output_index.front();
          }();

          pair_vec.emplace_back(task.inputs[alias.parameter_number],
                                task.outputs[output_number]);
        });

    stores.push_back(std::move(pair_vec));
  }
  return stores;
}

void Summarize(const std::vector<SpmdHloModuleTask>& tasks) {
  for (const auto& task : tasks) {
    std::cerr << task.module->module->name()
              << ",devices=" << task.device_assignment
              << ",loop=" << std::boolalpha << task.loop << std::endl;
    for (const auto& input : task.inputs) {
      std::cerr << task.module->module->name() << " has input " << input.name
                << ",index=" << input.index << ",type=" << (int)input.type
                << ",sharding=" << input.mpmd_sharding
                << ",shape=" << input.shape << std::endl;
    }
    for (const auto& output : task.outputs) {
      std::cerr << task.module->module->name() << " has output " << output.name
                << ",index=" << output.index << ",type=" << (int)output.type
                << ",sharding=" << output.mpmd_sharding
                << ",shape=" << output.shape << std::endl;
    }
  }
}

class MpmdPartitionTest : public MpmdTestBase {
 public:
  absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>> GetTempInOutPair(
      const HloModule& producer, const HloModule& consumer) {
    std::vector<HloInstruction*> roots;
    if (producer.entry_computation()->root_instruction()->shape().IsTuple()) {
      const auto& operands =
          producer.entry_computation()->root_instruction()->operands();
      roots = {operands.begin(), operands.end()};
    } else {
      roots.push_back(producer.entry_computation()->root_instruction());
    }

    for (auto* root : roots) {
      for (auto* param :
           consumer.entry_computation()->parameter_instructions()) {
        if (root->name() == param->name()) {
          return std::make_pair(root, param);
        }
      }
    }
    return InvalidArgumentStrCat(
        "modules have no shared in/out temporary pair");
  }
};

MATCHER(TaskNameStartsWith, "") {
  const auto& task = std::get<0>(arg);
  const auto& name = std::get<1>(arg);
  return absl::StartsWith(task.module->module->name(), name);
}

constexpr absl::string_view kSimpleMpmdHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[8]{0}, s32[])->(f32[], f32[8]{0})}, allow_spmd_sharding_propagation_to_parameters={true,true}, allow_spmd_sharding_propagation_to_output={true,true}

f.impl.12 {
  Arg_0.13 = f32[8]{0} parameter(0)
  multiply.15 = f32[8]{0} multiply(Arg_0.13, Arg_0.13)
  Arg_1.14 = s32[] parameter(1)
  convert.16 = f32[] convert(Arg_1.14)
  broadcast.17 = f32[8]{0} broadcast(convert.16), dimensions={}
  ROOT multiply.18 = f32[8]{0} multiply(multiply.15, broadcast.17)
} // f.impl.12

region_0.32 {
  Arg_0.33 = f32[] parameter(0)
  Arg_1.34 = f32[] parameter(1)
  ROOT add.35 = f32[] add(Arg_0.33, Arg_1.34)
}

g.impl.36 {
  Arg_0.37 = f32[8]{0} parameter(0)
  Arg_1.38 = f32[8]{0} parameter(1)
  add.41 = f32[8]{0} add(Arg_0.37, Arg_1.38)
  constant.39 = f32[] constant(0)
  ROOT reduce.42 = f32[] reduce(add.41, constant.39), dimensions={0}, to_apply=region_0.32
} // g.impl.36

g_bwd.impl.56 {
  Arg_1.58 = f32[8]{0} parameter(1)
  Arg_2.59 = f32[] parameter(2)
  broadcast.61 = f32[8]{0} broadcast(Arg_2.59), dimensions={}
  Arg_0.57 = f32[8]{0} parameter(0)
  sine.60 = f32[8]{0} sine(Arg_0.57), metadata={op_name="transpose(jvp"}
  multiply.63 = f32[8]{0} multiply(broadcast.61, sine.60)
  ROOT tuple.64 = (f32[8]{0}, f32[8]{0}) tuple(multiply.63, broadcast.61)
} // g_bwd.impl.56

f_bwd.impl_0.83 {
  Arg_2.86 = f32[8]{0} parameter(2)
  Arg_1.85 = s32[] parameter(1)
  convert.88 = f32[] convert(Arg_1.85)
  broadcast.89 = f32[8]{0} broadcast(convert.88), dimensions={}
  multiply.90 = f32[8]{0} multiply(Arg_2.86, broadcast.89)
  Arg_0.84 = f32[8]{0} parameter(0)
  multiply.92 = f32[8]{0} multiply(multiply.90, Arg_0.84), metadata={op_name="transpose(jvp"}
  multiply.91 = f32[8]{0} multiply(Arg_0.84, multiply.90)
  add.93 = f32[8]{0} add(multiply.92, multiply.91)
  constant.87 = pred[] constant(false)
  ROOT tuple.94 = (f32[8]{0}, pred[]) tuple(add.93, constant.87)
} // f_bwd.impl_0.83

ENTRY main.100 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={maximal device=0}
  Arg_1.2 = s32[] parameter(1), sharding={maximal device=0}
  custom-call.19 = f32[8]{0} custom-call(Arg_0.1, Arg_1.2), custom_call_target="MultiMeshTask", called_computations={f.impl.12}, backend_config={"name": "f", "devices": [0], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  custom-call.43 = f32[] custom-call(custom-call.19, Arg_0.1), custom_call_target="MultiMeshTask", called_computations={g.impl.36}, backend_config={"name": "g", "devices": [1], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  constant.3 = f32[] constant(1)
  custom-call.65 = (f32[8]{0}, f32[8]{0}) custom-call(custom-call.19, Arg_0.1, constant.3), custom_call_target="MultiMeshTask", called_computations={g_bwd.impl.56}, backend_config={"name": "bwd.g", "devices": [1], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  get-tuple-element.67 = f32[8]{0} get-tuple-element(custom-call.65), index=1
  get-tuple-element.66 = f32[8]{0} get-tuple-element(custom-call.65), index=0
  custom-call.95 = (f32[8]{0}, pred[]) custom-call(Arg_0.1, Arg_1.2, get-tuple-element.66), custom_call_target="MultiMeshTask", called_computations={f_bwd.impl_0.83}, backend_config={"name": "bwd.f", "devices": [0], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  get-tuple-element.96 = f32[8]{0} get-tuple-element(custom-call.95), index=0
  add.98 = f32[8]{0} add(get-tuple-element.67, get-tuple-element.96)
  ROOT tuple.99 = (f32[], f32[8]{0}) tuple(custom-call.43, add.98)
} // main.100
)";
static constexpr std::array kSimpleTaskOrder = {"f", "g", "bwd.g", "bwd.f"};

TEST_F(MpmdPartitionTest, SimpleMpmd) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, RunMpmdOnHloString(kSimpleMpmdHlo, /*num_devices=*/2));
  auto [tasks, intermediates] = std::move(result);

  EXPECT_THAT(tasks, Pointwise(TaskNameStartsWith(), kSimpleTaskOrder));

  EXPECT_THAT(
      tasks,
      ElementsAre(
          AllOf(m::TaskInputs(UnorderedElementsAre(
                    m::Store({.type = Store::Type::PARAM, .index = 0}),
                    m::Store({.type = Store::Type::PARAM, .index = 1}))),
                m::TaskOutputs(
                    ElementsAre(m::Store({.type = Store::Type::TEMP})))),
          // the inputs should be a resharded argument and resharded of temp
          // index=0, output should be a temp that gets resharded into root
          AllOf(m::TaskInputs(UnorderedElementsAre(
                    m::Store({.type = Store::Type::TEMP}),
                    m::Store({.type = Store::Type::TEMP}))),
                m::TaskOutputs(
                    ElementsAre(m::Store({.type = Store::Type::TEMP})))),
          AllOf(
              m::TaskInputs(ElementsAre(m::Store({.type = Store::Type::TEMP}))),
              // the buffer from the previous task should be reused
              m::TaskOutputs(
                  ElementsAre(m::Store({.type = Store::Type::TEMP})))),
          AllOf(m::TaskInputs(UnorderedElementsAre(
                    // the temp buffer from the earlier task should be reused
                    m::Store({.type = Store::Type::TEMP}),
                    m::Store({.type = Store::Type::PARAM, .index = 0}),
                    m::Store({.type = Store::Type::PARAM, .index = 1}))),
                m::TaskOutputs(ElementsAre(
                    m::Store({.type = Store::Type::ROOT, .index = 1}))))));
}

constexpr absl::string_view kUserSpecifiedShardingsHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[8]{0}, s32[])->(f32[], f32[8]{0})}, allow_spmd_sharding_propagation_to_parameters={true,false}, allow_spmd_sharding_propagation_to_output={false,true}

f.impl.12 {
  Arg_0.13 = f32[8]{0} parameter(0), metadata={op_name="jit(c)/jit(main)/mm_task"}
  multiply.15 = f32[8]{0} multiply(Arg_0.13, Arg_0.13), metadata={op_name="jit(c)/jit(main)/jvp(jit(f.impl))/mul"}
  Arg_1.14 = s32[] parameter(1), metadata={op_name="jit(c)/jit(main)/mm_task"}
  convert.16 = f32[] convert(Arg_1.14), metadata={op_name="jit(c)/jit(main)/jvp(jit(f.impl))/convert_element_type[new_dtype=float32 weak_type=False sharding=None]"}
  broadcast.17 = f32[8]{0} broadcast(convert.16), dimensions={}, metadata={op_name="jit(c)/jit(main)/jvp(jit(f.impl))/mul"}
  ROOT multiply.18 = f32[8]{0} multiply(multiply.15, broadcast.17), metadata={op_name="jit(c)/jit(main)/jvp(jit(f.impl))/mul"}
} // f.impl.12

region_0.32 {
  Arg_0.33 = f32[] parameter(0), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/reduce_sum[axes=(0,)]"}
  Arg_1.34 = f32[] parameter(1), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/reduce_sum[axes=(0,)]"}
  ROOT add.35 = f32[] add(Arg_0.33, Arg_1.34), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/reduce_sum[axes=(0,)]"}
}

g.impl.36 {
  Arg_0.37 = f32[8]{0} parameter(0), metadata={op_name="jit(c)/jit(main)/mm_task"}
  exp.40 = f32[8]{0} exponential(Arg_0.37), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/cos"}
  Arg_1.38 = f32[8]{0} parameter(1), metadata={op_name="jit(c)/jit(main)/mm_task"}
  add.41 = f32[8]{0} add(exp.40, Arg_1.38), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/add"}
  constant.39 = f32[] constant(0)
  ROOT reduce.42 = f32[] reduce(add.41, constant.39), dimensions={0}, to_apply=region_0.32, metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/reduce_sum[axes=(0,)]"}
} // g.impl.36

g_bwd.impl.56 {
  Arg_1.58 = f32[8]{0} parameter(1)
  Arg_2.59 = f32[] parameter(2)
  broadcast.61 = f32[8]{0} broadcast(Arg_2.59), dimensions={}
  negate.62 = f32[8]{0} negate(broadcast.61)
  Arg_0.57 = f32[8]{0} parameter(0)
  sine.60 = f32[8]{0} sine(Arg_0.57)
  multiply.63 = f32[8]{0} multiply(negate.62, sine.60)
  ROOT tuple.64 = (f32[8]{0}, f32[8]{0}) tuple(multiply.63, broadcast.61)
} // g_bwd.impl.56

f_bwd.impl_0.83 {
  Arg_2.86 = f32[8]{0} parameter(2)
  Arg_1.85 = s32[] parameter(1)
  convert.88 = f32[] convert(Arg_1.85)
  broadcast.89 = f32[8]{0} broadcast(convert.88), dimensions={}
  multiply.90 = f32[8]{0} multiply(Arg_2.86, broadcast.89)
  Arg_0.84 = f32[8]{0} parameter(0)
  multiply.92 = f32[8]{0} multiply(multiply.90, Arg_0.84)
  multiply.91 = f32[8]{0} multiply(Arg_0.84, multiply.90)
  add.93 = f32[8]{0} add(multiply.92, multiply.91)
  constant.87 = pred[] constant(false)
  ROOT tuple.94 = (f32[8]{0}, pred[]) tuple(add.93, constant.87)
} // f_bwd.impl_0.83

ENTRY main.100 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  Arg_1.2 = s32[] parameter(1), sharding={replicated}
  custom-call.19 = f32[8]{0} custom-call(Arg_0.1, Arg_1.2), custom_call_target="MultiMeshTask", called_computations={f.impl.12}, backend_config={"name": "f", "devices": [0], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  custom-call.43 = f32[] custom-call(custom-call.19, Arg_0.1), custom_call_target="MultiMeshTask", called_computations={g.impl.36}, backend_config={"name": "g", "devices": [1], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  constant.3 = f32[] constant(1)
  custom-call.65 = (f32[8]{0}, f32[8]{0}) custom-call(custom-call.19, Arg_0.1, constant.3), custom_call_target="MultiMeshTask", called_computations={g_bwd.impl.56}, backend_config={"name": "g.bwd", "devices": [1], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  get-tuple-element.67 = f32[8]{0} get-tuple-element(custom-call.65), index=1
  get-tuple-element.66 = f32[8]{0} get-tuple-element(custom-call.65), index=0
  custom-call.95 = (f32[8]{0}, pred[]) custom-call(Arg_0.1, Arg_1.2, get-tuple-element.66), custom_call_target="MultiMeshTask", called_computations={f_bwd.impl_0.83}, backend_config={"name": "f.bwd", "devices": [0], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  get-tuple-element.96 = f32[8]{0} get-tuple-element(custom-call.95), index=0
  add.98 = f32[8]{0} add(get-tuple-element.67, get-tuple-element.96)
  ROOT tuple.99 = (f32[], f32[8]{0}) tuple(custom-call.43, add.98), sharding={{replicated},{devices=[2]<=[2]}}
} // main.100
)";

TEST_F(MpmdPartitionTest, UserSpecifiedShardings) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kUserSpecifiedShardingsHlo, /*num_devices=*/2,
                         {.use_module_config_auto_param_sharding = true,
                          .use_module_config_auto_output_sharding = true}));
  auto [tasks, intermediates] = std::move(result);
}

static constexpr absl::string_view kShardingPropagationIntermediateHlo = R"(
f.impl_0 {
  Arg_0.1 = f32[8]{0} parameter(0)
  constant.2 = f32[] constant(2)
  broadcast.3 = f32[8]{0} broadcast(constant.2), dimensions={}
  ROOT add.6 = f32[8]{0} add(Arg_0.1, broadcast.3)
}

g.impl_0 {
  Arg_0.1 = f32[8]{0} parameter(0)
  cosine.11 = f32[8]{0} cosine(Arg_0.1)
  constant.2 = f32[] constant(2)
  broadcast.3 = f32[8]{0} broadcast(constant.2), dimensions={}
  ROOT multiply.12 = f32[8]{0} multiply(cosine.11, broadcast.3)
}

ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={devices=[2]<=[2]}
  custom-call.4 = f32[8]{0} custom-call(Arg_0.1), custom_call_target="MultiMeshTask", called_computations={f.impl_0}, backend_config={"type": "input", "name": "f", "devices": [0, 1]}
  ROOT custom-call.7 = f32[8]{0} custom-call(custom-call.4), custom_call_target="MultiMeshTask", called_computations={g.impl_0}, backend_config={"type": "output", "name": "g", "devices": [2, 3]}
} // main.15
)";
static constexpr std::array kShardingPropagationIntermediateTaskOrder = {"f",
                                                                         "g"};
static constexpr absl::string_view
    kShardingPropagationIntermediateShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 2
iota_reshape_dims: 2
iota_transpose_perm: 0
)";

TEST_F(MpmdPartitionTest, ShardingPropagationIntermediate) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, RunMpmdOnHloString(kShardingPropagationIntermediateHlo,
                                      /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  TF_ASSERT_OK_AND_ASSIGN(
      HloSharding expected_sharding,
      GetSharding(kShardingPropagationIntermediateShardingPbtxt));

  EXPECT_THAT(tasks, Pointwise(TaskNameStartsWith(),
                               kShardingPropagationIntermediateTaskOrder));

  ASSERT_EQ(tasks.size(), 2);

  auto* f0_root =
      tasks[0].module->module->entry_computation()->root_instruction();
  EXPECT_THAT(f0_root->operand(0), op::Sharding(expected_sharding));

  ASSERT_EQ(tasks[0].module->module->entry_computation()->num_parameters(), 1);

  EXPECT_THAT(
      tasks[0].module->module->entry_computation()->parameter_instructions(),
      ElementsAre(op::Sharding(expected_sharding)));
}

static constexpr absl::string_view kMjitTaskReshardingHlo = R"(
f.impl_0 {
  Arg_0.1 = f32[4,2]{1,0} parameter(0)
  custom-call.7 = f32[4,2]{1,0} custom-call(Arg_0.1), custom_call_target="Sharding", sharding={devices=[2,1]<=[2]}
  constant.3 = f32[] constant(2)
  broadcast.4 = f32[4,2]{1,0} broadcast(constant.3), dimensions={}
  ROOT add.8 = f32[4,2]{1,0} add(custom-call.7, broadcast.4)
}

g.impl_0 {
  Arg_0.1 = f32[4,2]{1,0} parameter(0)
  Arg_1.2 = f32[4,2]{1,0} parameter(1)
  cosine.19 = f32[4,2]{1,0} cosine(Arg_0.1)
  ROOT multiply.20 = f32[4,2]{1,0} multiply(cosine.19, Arg_1.2)
}

ENTRY main.23 {
  Arg_0.1 = f32[4,2]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  custom-call.4 = f32[4,2]{1,0} custom-call(Arg_0.1), custom_call_target="MultiMeshTask", called_computations={f.impl_0}, backend_config={"type": "input", "name": "f", "devices": [0, 1]}
  custom-call.17 = f32[4,2]{1,0} custom-call(custom-call.4), custom_call_target="Sharding", sharding={replicated}
  Arg_1.2 = f32[4,2]{1,0} parameter(1), sharding={devices=[2,1]<=[2]}
  ROOT custom-call.5 = f32[4,2]{1,0} custom-call(custom-call.17, Arg_1.2), custom_call_target="MultiMeshTask", called_computations={g.impl_0}, backend_config={"type": "input", "name": "g", "devices": [2, 3]}
}

)";
static constexpr absl::string_view kMjitTaskReshardingShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 1
iota_reshape_dims: 2
iota_transpose_perm: 0
)";
static constexpr absl::string_view kReplicatedShardingPbtxt = R"(
type: REPLICATED
)";

TEST_F(MpmdPartitionTest, MjitTaskResharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kMjitTaskReshardingHlo, /*num_devices=*/2));
  auto [tasks, intermediates] = std::move(result);

  ASSERT_EQ(tasks.size(), 2);

  TF_ASSERT_OK_AND_ASSIGN(HloSharding tiled_sharding,
                          GetSharding(kMjitTaskReshardingShardingPbtxt));

  // The temporary parameter between these tasks should switch from sharded to
  // replicated
  TF_ASSERT_OK_AND_ASSIGN(HloSharding replicated_sharding,
                          GetSharding(kReplicatedShardingPbtxt));

  EXPECT_THAT(
      tasks[0].module->module->entry_computation()->root_instruction()->operand(
          0),
      op::Sharding(replicated_sharding));

  EXPECT_THAT(
      tasks[1].module->module->entry_computation()->parameter_instructions(),
      ElementsAre(AnyOf(op::NoSharding(), op::Sharding(replicated_sharding)),
                  op::Sharding(tiled_sharding)));

  // The output of the second tasks should be sharded again
  EXPECT_THAT(
      tasks[1].module->module->entry_computation()->root_instruction()->operand(
          0),
      op::Sharding(tiled_sharding));
}

static constexpr absl::string_view kConstantOutputHlo = R"(
HloModule jit_args_maker, entry_computation_layout={()->(f32[8]{0}, s32[], s32[])}, allow_spmd_sharding_propagation_to_output={true,true,true}

ENTRY main.4 {
  iota.2 = f32[8]{0} iota(), iota_dimension=0
  constant.1 = s32[] constant(2)
  ROOT tuple.3 = (f32[8]{0}, s32[], s32[]) tuple(iota.2, constant.1, constant.1)
}
)";

TEST_F(MpmdPartitionTest, ConstantOutput) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kConstantOutputHlo, /*num_devices=*/1,
                         {.use_module_config_auto_output_sharding = true}));

  auto [tasks, intermediates] = std::move(result);

  // single task with 3 outputs
  EXPECT_THAT(tasks, ElementsAre(NumOutputs(3)));
}
static constexpr absl::string_view kPreambleTasks = R"(
region_0.27 {
  Arg_0.28 = f32[] parameter(0)
  Arg_1.29 = f32[] parameter(1)
  ROOT add.30 = f32[] add(Arg_0.28, Arg_1.29)
}

f.impl_0 {
  Arg_0.1 = f32[8]{0} parameter(0)
  Arg_1.2 = s32[] parameter(1)
  convert.14 = f32[] convert(Arg_1.2)
  broadcast.15 = f32[8]{0} broadcast(convert.14), dimensions={}
  ROOT multiply.16 = f32[8]{0} multiply(Arg_0.1, broadcast.15)
}

g.impl_0 {
  Arg_0.1 = f32[8]{0} parameter(0)
  Arg_1.2 = s32[] parameter(1)
  Arg_2.3 = f32[8]{0} parameter(2)
  convert.14 = f32[] convert(Arg_1.2)
  broadcast.15 = f32[8]{0} broadcast(convert.14), dimensions={}
  cosine.25 = f32[8]{0} cosine(Arg_2.3)
  ROOT add.26 = f32[8]{0} add(cosine.25, Arg_0.1)
}

ENTRY main.34 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  constant.3 = f32[] constant(2)
  broadcast.4 = f32[8]{0} broadcast(constant.3), dimensions={}
  multiply.6 = f32[8]{0} multiply(Arg_0.1, broadcast.4)
  Arg_1.2 = s32[] parameter(1), sharding={replicated}
  Arg_2.3 = f32[8]{0} parameter(2), sharding={replicated}
  custom-call.4 = f32[8]{0} custom-call(multiply.6, Arg_1.2), custom_call_target="MultiMeshTask", called_computations={f.impl_0}, backend_config={"type": "input", "name": "f", "devices": [0, 1]}
  ROOT custom-call.5 = f32[8]{0} custom-call(custom-call.4, Arg_1.2, Arg_2.3), custom_call_target="MultiMeshTask", called_computations={g.impl_0}, backend_config={"type": "input", "name": "f", "devices": [0, 1]}
} // main.34
)";

TEST_F(MpmdPartitionTest, PreambleInstructions) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kPreambleTasks, /*num_devices=*/2,
                         {.use_module_config_auto_output_sharding = true}));
  auto [tasks, intermediates] = std::move(result);

  // just validate that the decomposition worked
  // all of the preamble instructions should have gotten pulled into
  // one of the two tasks
  EXPECT_EQ(tasks.size(), 2);
}

static constexpr absl::string_view kIntermediateTupleHlo = R"(
ENTRY %main.7 (Arg_0.1: s32[2,2,2]) -> s32[2,2,2] {
  %Arg_0.1 = s32[2,2,2]{2,1,0} parameter(0), sharding={replicated}
  %constant.2 = s32[] constant(2)
  %broadcast.3 = s32[2,2,2]{2,1,0} broadcast(s32[] %constant.2), dimensions={}
  %multiply.4 = s32[2,2,2]{2,1,0} multiply(s32[2,2,2]{2,1,0} %Arg_0.1, s32[2,2,2]{2,1,0} %broadcast.3)
  %tuple.5 = (s32[2,2,2]{2,1,0}) tuple(s32[2,2,2]{2,1,0} %multiply.4)
  ROOT %get-tuple-element.6 = s32[2,2,2]{2,1,0} get-tuple-element((s32[2,2,2]{2,1,0}) %tuple.5), index=0, sharding={replicated}
}
)";

TEST_F(MpmdPartitionTest, IntermediateTuple) {
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunMpmdOnHloString(kIntermediateTupleHlo,
                                                          /*num_devices*/ 1));
  auto [tasks, intermediates] = std::move(result);
  // should be a single task
  EXPECT_EQ(tasks.size(), 1);
  // and there should be no intermediates
  EXPECT_EQ(intermediates, 0);
}

static constexpr absl::string_view kShardedGetTupleElementAliasRootHlo = R"(
HloModule test, entry_computation_layout={(f32[4])->f32[4]}, allow_spmd_sharding_propagation_to_output={false}

ENTRY main.62 {
  Arg_0.1 = f32[4] parameter(0), sharding={devices=[2]<=[2]}
  add.44 = f32[4] add(Arg_0.1,Arg_0.1)
  tuple.6 = (f32[4]) tuple(add.44)
  ROOT get-tuple-element.59 = f32[4] get-tuple-element(tuple.6), index=0, sharding={replicated}
} // main.62
)";

TEST_F(MpmdPartitionTest, ShardedGetTupleElementAliasRoot) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kShardedGetTupleElementAliasRootHlo, /*num_devices=*/2,
                         {.use_module_config_auto_output_sharding = true}));
  auto [tasks, intermediates] = std::move(result);

  EXPECT_THAT(tasks, ElementsAre(m::TaskOutputs(Each(
                         m::Store({.sharding = HloSharding::Replicate()})))));
}

static constexpr absl::string_view kMicrobatchMultipleTasksInLoopHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[4,4]{1,0}, f32[4]{0}, f32[4]{0})->f32[]}, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={true}

f.impl.17.clone {
  Arg_0.0 = f32[2,4]{1,0} parameter(0), metadata={op_name="jit(c)/jit(main)/while/body/mm_task"}
  Arg_1.0 = f32[4]{0} parameter(1), metadata={op_name="jit(c)/jit(main)/while/body/mm_task"}
  reshape.0 = f32[1,4]{1,0} reshape(Arg_1.0), metadata={op_name="jit(c)/jit(main)/while/body/jit(f.impl)/broadcast_in_dim[shape=(1, 4) broadcast_dimensions=(1,)]"}
  broadcast.0 = f32[1,4]{1,0} broadcast(reshape.0), dimensions={0,1}, metadata={op_name="jit(c)/jit(main)/while/body/jit(f.impl)/mul"}
  reshape.1 = f32[4]{0} reshape(broadcast.0), metadata={op_name="jit(c)/jit(main)/while/body/jit(f.impl)/mul"}
  broadcast.1 = f32[2,4]{1,0} broadcast(reshape.1), dimensions={1}, metadata={op_name="jit(c)/jit(main)/while/body/jit(f.impl)/mul"}
  ROOT multiply.0 = f32[2,4]{1,0} multiply(Arg_0.0, broadcast.1), metadata={op_name="jit(c)/jit(main)/while/body/jit(f.impl)/mul"}
} // f.impl.17.clone

region_1.39 {
  Arg_0.40 = f32[] parameter(0), metadata={op_name="jit(c)/jit(main)/while/body/jit(g.impl)/reduce_sum[axes=(0, 1)]"}
  Arg_1.41 = f32[] parameter(1), metadata={op_name="jit(c)/jit(main)/while/body/jit(g.impl)/reduce_sum[axes=(0, 1)]"}
  ROOT add.42 = f32[] add(Arg_0.40, Arg_1.41), metadata={op_name="jit(c)/jit(main)/while/body/jit(g.impl)/reduce_sum[axes=(0, 1)]"}
}

g.impl.43.clone {
  Arg_0.2 = f32[2,4]{1,0} parameter(0), metadata={op_name="jit(c)/jit(main)/while/body/mm_task"}
  Arg_1.1 = f32[4]{0} parameter(1), metadata={op_name="jit(c)/jit(main)/while/body/mm_task"}
  reshape.2 = f32[1,4]{1,0} reshape(Arg_1.1), metadata={op_name="jit(c)/jit(main)/while/body/jit(g.impl)/broadcast_in_dim[shape=(1, 4) broadcast_dimensions=(1,)]"}
  broadcast.2 = f32[1,4]{1,0} broadcast(reshape.2), dimensions={0,1}, metadata={op_name="jit(c)/jit(main)/while/body/jit(g.impl)/mul"}
  reshape.3 = f32[4]{0} reshape(broadcast.2), metadata={op_name="jit(c)/jit(main)/while/body/jit(g.impl)/mul"}
  broadcast.3 = f32[2,4]{1,0} broadcast(reshape.3), dimensions={1}, metadata={op_name="jit(c)/jit(main)/while/body/jit(g.impl)/mul"}
  multiply.1 = f32[2,4]{1,0} multiply(Arg_0.2, broadcast.3), metadata={op_name="jit(c)/jit(main)/while/body/jit(g.impl)/mul"}
  constant.3 = f32[] constant(0)
  ROOT reduce.0 = f32[] reduce(multiply.1, constant.3), dimensions={0,1}, to_apply=region_1.39, metadata={op_name="jit(c)/jit(main)/while/body/jit(g.impl)/reduce_sum[axes=(0, 1)]"}
} // g.impl.43.clone

region_0.74 {
  arg_tuple.75 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4]{0}, /*index=5*/f32[4]{0}) parameter(0)
  get-tuple-element.76 = s32[] get-tuple-element(arg_tuple.75), index=0
  constant.82 = s32[] constant(1)
  add.86 = s32[] add(get-tuple-element.76, constant.82), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  get-tuple-element.77 = s32[] get-tuple-element(arg_tuple.75), index=1
  constant.0 = s32[] constant(2)
  add.0 = s32[] add(get-tuple-element.77, constant.0), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  get-tuple-element.78 = f32[] get-tuple-element(arg_tuple.75), index=2
  get-tuple-element.79 = f32[4,4]{1,0} get-tuple-element(arg_tuple.75), index=3
  constant.1 = s32[] constant(0)
  compare.0 = pred[] compare(get-tuple-element.77, constant.1), direction=LT, metadata={op_name="jit(c)/jit(main)/while/body/lt"}
  constant.2 = s32[] constant(4)
  add.1 = s32[] add(get-tuple-element.77, constant.2), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  select.0 = s32[] select(compare.0, add.1, get-tuple-element.77), metadata={op_name="jit(c)/jit(main)/while/body/select_n"}
  dynamic-slice.0 = f32[2,4]{1,0} dynamic-slice(get-tuple-element.79, select.0, constant.1), dynamic_slice_sizes={2,4}, metadata={op_name="jit(c)/jit(main)/while/body/dynamic_slice[slice_sizes=(2, 4)]"}
  custom-call.0 = f32[2,4]{1,0} custom-call(dynamic-slice.0), custom_call_target="MicrobatchSlice", metadata={op_name="jit(c)/jit(main)/while/body/MicrobatchSlice/MicrobatchSlice"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  get-tuple-element.80 = f32[4]{0} get-tuple-element(arg_tuple.75), index=4
  custom-call.1 = f32[2,4]{1,0} custom-call(custom-call.0, get-tuple-element.80), custom_call_target="MultiMeshTask", called_computations={f.impl.17.clone}, metadata={op_name="jit(c)/jit(main)/while/body/mm_task"}, backend_config={"name": "f", "devices": [0,1], "color": 0, "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  get-tuple-element.81 = f32[4]{0} get-tuple-element(arg_tuple.75), index=5
  custom-call.2 = f32[] custom-call(custom-call.1, get-tuple-element.81), custom_call_target="MultiMeshTask", called_computations={g.impl.43.clone}, metadata={op_name="jit(c)/jit(main)/while/body/mm_task"}, backend_config={"name": "g", "devices": [2,3], "color": 0, "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  add.2 = f32[] add(get-tuple-element.78, custom-call.2), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  ROOT tuple.87 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4]{0}, /*index=5*/f32[4]{0}) tuple(add.86, add.0, add.2, get-tuple-element.79, get-tuple-element.80, /*index=5*/get-tuple-element.81)
} // region_0.74

region_2.88 {
  arg_tuple.89 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4]{0}, /*index=5*/f32[4]{0}) parameter(0)
  get-tuple-element.90 = s32[] get-tuple-element(arg_tuple.89), index=0
  constant.96 = s32[] constant(2)
  ROOT compare.97 = pred[] compare(get-tuple-element.90, constant.96), direction=LT, metadata={op_name="jit(c)/jit(main)/while/cond/lt"}
}

ENTRY main.105 {
  constant.4 = s32[] constant(0)
  constant.5 = f32[] constant(0)
  custom-call.6 = f32[] custom-call(constant.5), custom_call_target="MicrobatchInit", metadata={op_name="jit(c)/jit(main)/MicrobatchInit/MicrobatchInit"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  Arg_0.1 = f32[4,4]{1,0} parameter(0), metadata={op_name="x"}
  custom-call.7 = f32[4,4]{1,0} custom-call(Arg_0.1), custom_call_target="Microbatch", metadata={op_name="jit(c)/jit(main)/Microbatch/Microbatch"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  Arg_1.2 = f32[4]{0} parameter(1), metadata={op_name="param1"}
  Arg_2.3 = f32[4]{0} parameter(2), metadata={op_name="param2"}
  tuple.8 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4]{0}, /*index=5*/f32[4]{0}) tuple(constant.4, constant.4, custom-call.6, custom-call.7, Arg_1.2, /*index=5*/Arg_2.3), metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  while.98 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4]{0}, /*index=5*/f32[4]{0}) while(tuple.8), condition=region_2.88, body=region_0.74, metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  ROOT get-tuple-element.101 = f32[] get-tuple-element(while.98), index=2, metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
} // main.105
)";

TEST_F(MpmdPartitionTest, MicrobatchMultipleTasksInLoopHlo) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kMicrobatchMultipleTasksInLoopHlo, /*num_devices=*/2,
                         {.use_module_config_auto_output_sharding = true}));

  auto [tasks, intermediates] = std::move(result);

  // there should be a preamble task before the microbatching
  // and two loop-dependent tasks after the microbatching
  EXPECT_THAT(tasks, UnorderedElementsAre(Not(m::LoopTask()), m::LoopTask(true),
                                          m::LoopTask(true), m::LoopTask(false),
                                          m::LoopTask(false)));

  EXPECT_THAT(
      tasks, UnorderedElementsAre(
                 // initialize loop accumulator
                 Field(&SpmdHloModuleTask::outputs,
                       IsSupersetOf({m::Store({.type = Store::Type::ROOT})})),
                 // microbatch 0 output
                 Field(&SpmdHloModuleTask::outputs,
                       ElementsAre(m::Store({.type = Store::Type::TEMP}))),
                 // microbatch 1 output
                 Field(&SpmdHloModuleTask::outputs,
                       ElementsAre(m::Store({.type = Store::Type::TEMP}))),
                 // microbatch 0 accumulation
                 Field(&SpmdHloModuleTask::outputs,
                       IsSupersetOf({m::Store({.type = Store::Type::ROOT})})),
                 // microbatch 1 accumulation
                 Field(&SpmdHloModuleTask::outputs,
                       IsSupersetOf({m::Store({.type = Store::Type::ROOT})}))));

  EXPECT_THAT(
      tasks,
      UnorderedElementsAre(
          // init accumulator
          Field(&SpmdHloModuleTask::inputs, ElementsAre(/*empty*/)),
          // microbatch 0 output
          Field(&SpmdHloModuleTask::inputs,  // one task should take two params
                                             // and the slice offset
                IsSupersetOf({m::Store({.type = Store::Type::TEMP}),
                              m::Store({.type = Store::Type::PARAM}),
                              m::Store({.type = Store::Type::PARAM})})),
          // microbatch 1 output
          Field(&SpmdHloModuleTask::inputs,  // one task should take two params
                                             // and the slice offset
                IsSupersetOf({m::Store({.type = Store::Type::TEMP}),
                              m::Store({.type = Store::Type::PARAM}),
                              m::Store({.type = Store::Type::PARAM})})),
          // accumulate microbatch 0
          Field(&SpmdHloModuleTask::inputs,  // the final loop task should take
                                             // a resharded param and an
                                             // intermediate and the loop
                                             // accumulator
                IsSupersetOf({m::Store({.type = Store::Type::ROOT}),
                              m::Store({.type = Store::Type::PARAM}),
                              m::Store({.type = Store::Type::TEMP})})),
          // accumulate microbatch 1
          Field(&SpmdHloModuleTask::inputs,  // the final loop task should take
                                             // a resharded param and an
                                             // intermediate and the loop
                                             // accumulator
                IsSupersetOf({m::Store({.type = Store::Type::ROOT}),
                              m::Store({.type = Store::Type::PARAM}),
                              m::Store({.type = Store::Type::TEMP})}))));
}

static constexpr absl::string_view kSimpleImplicitTaskHlo = R"(
region_0.10 {
  Arg_0.11 = f32[] parameter(0)
  Arg_1.12 = f32[] parameter(1)
  ROOT add.13 = f32[] add(Arg_0.11, Arg_1.12), metadata={op_name="jit(c)/jit(main)/task_g/reduce_sum[axes=(0,)]"}
}

ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  multiply.4 = f32[8]{0} multiply(Arg_0.1, Arg_0.1), metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  Arg_1.2 = s32[] parameter(1), sharding={replicated}
  convert.5 = f32[] convert(Arg_1.2), metadata={op_name="jit(c)/jit(main)/task_f/convert_element_type[new_dtype=float32 weak_type=False]"}
  broadcast.6 = f32[8]{0} broadcast(convert.5), dimensions={}, metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  multiply.7 = f32[8]{0} multiply(multiply.4, broadcast.6), metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  cosine.8 = f32[8]{0} cosine(multiply.7), metadata={op_name="jit(c)/jit(main)/task_g/cos"}
  add.9 = f32[8]{0} add(cosine.8, multiply.7), metadata={op_name="jit(c)/jit(main)/task_g/add"}
  constant.3 = f32[] constant(0)
  ROOT reduce.14 = f32[] reduce(add.9, constant.3), dimensions={0}, to_apply=region_0.10, metadata={op_name="jit(c)/jit(main)/task_g/reduce_sum[axes=(0,)]"}
} // main.15
)";

TEST_F(MpmdPartitionTest, SimpleImplicitTask) {
  RegisterNamedTestTask("task_f", {0, 2}, {2}, {"x"}, {{"x", "batch"}});
  RegisterNamedTestTask("task_g", {2, 4}, {2}, {"x"}, {{"x", "batch"}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kSimpleImplicitTaskHlo, /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  // there should be two tasks with a single temp intermediate between them
  EXPECT_THAT(
      tasks,
      ElementsAre(
          Field(&SpmdHloModuleTask::outputs,
                Contains(m::Store({.type = Store::Type::TEMP})).Times(1)),
          Field(&SpmdHloModuleTask::inputs,
                Contains(m::Store({.type = Store::Type::TEMP})).Times(1))));

  // the first task should go to device assignment 0,1
  // the second task should go to device assignment 2,3
  EXPECT_THAT(
      tasks,
      ElementsAre(
          Field(&SpmdHloModuleTask::device_assignment, ElementsAre(0, 1)),
          Field(&SpmdHloModuleTask::device_assignment, ElementsAre(2, 3))));
}

static constexpr absl::string_view kEmbedLayerLossComputationHlo = R"(
_one_hot.5 {
  Arg_0.6 = f32[4,2]{1,0} parameter(0)
  reshape.7 = f32[4,2,1]{2,1,0} reshape(Arg_0.6), metadata={op_name="embedding"}
  broadcast.10 = f32[4,2,1]{2,1,0} broadcast(reshape.7), dimensions={0,1,2}, metadata={op_name="embedding"}
  reshape.11 = f32[4,2]{1,0} reshape(broadcast.10), metadata={op_name="embedding"}
  broadcast.12 = f32[4,2,8]{2,1,0} broadcast(reshape.11), dimensions={0,1}, metadata={op_name="embedding"}
  iota.8 = f32[8]{0} iota(), iota_dimension=0, metadata={op_name="embedding"}
  reshape.9 = f32[1,1,8]{2,1,0} reshape(iota.8), metadata={op_name="embedding"}
  broadcast.13 = f32[1,1,8]{2,1,0} broadcast(reshape.9), dimensions={0,1,2}, metadata={op_name="embedding"}
  reshape.14 = f32[8]{0} reshape(broadcast.13), metadata={op_name="embedding"}
  broadcast.15 = f32[4,2,8]{2,1,0} broadcast(reshape.14), dimensions={2}, metadata={op_name="embedding"}
  compare.16 = pred[4,2,8]{2,1,0} compare(broadcast.12, broadcast.15), direction=EQ, metadata={op_name=""}
  ROOT convert.17 = f32[4,2,8]{2,1,0} convert(compare.16), metadata={op_name="embedding"}
} // _one_hot.5

region_0.27 {
  Arg_0.28 = f32[] parameter(0)
  Arg_1.29 = f32[] parameter(1)
  ROOT add.30 = f32[] add(Arg_0.28, Arg_1.29), metadata={op_name=""}
}

ENTRY main.32 {
  Arg_2.3 = f32[4,2]{1,0} parameter(2), sharding={devices=[4,1]<=[4]}
  call.18 = f32[4,2,8]{2,1,0} call(Arg_2.3), to_apply=_one_hot.5
  Arg_1.2 = f32[8,4]{1,0} parameter(1), sharding={replicated}
  custom-call.10 = f32[8,4]{1,0} custom-call(Arg_1.2), custom_call_target="AutoSharding", backend_config={"axes": [["x"], ["y"]]}
  dot.19 = f32[4,2,4]{2,1,0} dot(call.18, custom-call.10), lhs_contracting_dims={2}, rhs_contracting_dims={0}, metadata={op_name="embedding"}
  custom-call.12 = f32[4,2,4]{2,1,0} custom-call(dot.19), custom_call_target="AutoSharding", backend_config={"axes": [["x"], ["y"], ["other"]]}
  Arg_0.1 = f32[4]{0} parameter(0), sharding={replicated}
  custom-call.11 = f32[4]{0} custom-call(Arg_0.1), custom_call_target="AutoSharding", backend_config={"axes": [["x"]]}
  reshape.21 = f32[1,1,4]{2,1,0} reshape(custom-call.11), metadata={op_name="task_0"}
  broadcast.22 = f32[1,1,4]{2,1,0} broadcast(reshape.21), dimensions={0,1,2}, metadata={op_name=""}
  reshape.23 = f32[4]{0} reshape(broadcast.22), metadata={op_name="task_0"}
  broadcast.24 = f32[4,2,4]{2,1,0} broadcast(reshape.23), dimensions={2}, metadata={op_name="task_0"}
  multiply.25 = f32[4,2,4]{2,1,0} multiply(custom-call.12, broadcast.24), metadata={op_name="task_0"}
  multiply.26 = f32[4,2,4]{2,1,0} multiply(multiply.25, multiply.25), metadata={op_name="task_1"}
  constant.4 = f32[] constant(0)
  ROOT reduce.31 = f32[] reduce(multiply.26, constant.4), dimensions={0,1,2}, to_apply=region_0.27, metadata={op_name="task_1"}
} // main.32
)";
static constexpr absl::string_view kEmbedTempOutputSharding = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 2
tile_assignment_dimensions: 1
iota_reshape_dims: 4
iota_transpose_perm: 0
)";
static constexpr absl::string_view kEmbedTempInputSharding = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 1
tile_assignment_dimensions: 1
iota_reshape_dims: 2
iota_transpose_perm: 0
)";

TEST_F(MpmdPartitionTest, ReshardGlobalInputsToSubmeshInputs) {
  RegisterNamedTestTask("embedding", {0, 4}, {2, 2}, {"x", "y"},
                        {{"x", "x"}, {"y", "y"}});
  RegisterNamedTestTask("task_0", {0, 2}, {2, 1}, {"x", "y"},
                        {{"x", "x"}, {"y", "y"}});
  RegisterNamedTestTask("task_1", {2, 4}, {2, 1}, {"x", "y"},
                        {{"x", "x"}, {"y", "y"}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kEmbedLayerLossComputationHlo, /*num_devices=*/4,
                         {.use_auto_input_sharding = true}));
  auto [tasks, intermediates] = std::move(result);

  TF_ASSERT_OK_AND_ASSIGN(HloSharding embed_output_sharding,
                          GetSharding(kEmbedTempOutputSharding));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding task0_input_sharding,
                          GetSharding(kEmbedTempInputSharding));

  EXPECT_THAT(
      tasks,
      ElementsAre(
          Field(&SpmdHloModuleTask::outputs,
                Contains(m::Store({.index = 0,
                                   .type = Store::Type::TEMP,
                                   .sharding = embed_output_sharding}))),
          AllOf(Field(&SpmdHloModuleTask::inputs,
                      // reshard of output from first task
                      Contains(m::Store({.index = 1,
                                         .type = Store::Type::TEMP,
                                         .sharding = task0_input_sharding}))),
                Field(&SpmdHloModuleTask::outputs,
                      Contains(
                          m::Store({.index = 2, .type = Store::Type::TEMP})))),
          // reshard back to first mesh
          Field(&SpmdHloModuleTask::inputs,
                Contains(m::Store({.index = 3, .type = Store::Type::TEMP})))));
}

static constexpr absl::string_view kImplicitlySeparateShardingHlo = R"(
ENTRY %main.7 {
  %Arg_0.1 = s32[8]{0} parameter(0), sharding={replicated}
  %constant.2 = s32[] constant(2)
  %broadcast.3 = s32[8]{0} broadcast(s32[] %constant.2), dimensions={}
  %multiply.5 = s32[8]{0} multiply(s32[8]{0} %Arg_0.1, s32[8]{0} %broadcast.3)
  %multiply.6 = s32[8]{0} multiply(s32[8]{0} %Arg_0.1, s32[8]{0} %broadcast.3)
  ROOT %tuple.7 = (s32[8]{0},s32[8]{0}) tuple(s32[8]{0} %multiply.5, s32[8]{0} %multiply.6), sharding={{devices=[2]0,1},{devices=[2]2,3}}
}
)";
static constexpr absl::string_view kSeparateOutputSharding = R"(
type: OTHER
tile_assignment_dimensions: 2
iota_reshape_dims: 2
iota_transpose_perm: 0
)";

TEST_F(MpmdPartitionTest, ImplicitlySeparateSharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kImplicitlySeparateShardingHlo, /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  // This should produce two different tasks for each of the different root
  // shardings
  EXPECT_THAT(tasks, ElementsAre(m::TaskDevices(ElementsAre(0, 1)),
                                 m::TaskDevices(ElementsAre(2, 3))));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding output_sharding,
                          GetSharding(kSeparateOutputSharding));

  EXPECT_THAT(tasks,
              Each(m::TaskRoots(ElementsAre(op::Sharding(output_sharding)))));
}

static constexpr absl::string_view kImplicitlySameShardingHlo = R"(
ENTRY %main.7 {
  %Arg_0.1 = s32[8]{0} parameter(0), sharding={replicated}
  %constant.2 = s32[] constant(2)
  %broadcast.3 = s32[8]{0} broadcast(s32[] %constant.2), dimensions={}
  %multiply.5 = s32[8]{0} multiply(s32[8]{0} %Arg_0.1, s32[8]{0} %broadcast.3)
  %multiply.6 = s32[8]{0} multiply(s32[8]{0} %Arg_0.1, s32[8]{0} %broadcast.3)
  ROOT %tuple.7 = (s32[8]{0},s32[8]{0}) tuple(s32[8]{0} %multiply.5, s32[8]{0} %multiply.6), sharding={{devices=[2]0,1},{devices=[2]0,1}}
}
)";
static constexpr absl::string_view kSameOutputSharding = R"(
type: OTHER
tile_assignment_dimensions: 2
iota_reshape_dims: 2
iota_transpose_perm: 0
)";

TEST_F(MpmdPartitionTest, ImplicitlySameSharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kImplicitlySameShardingHlo, /*num_devices=*/2));
  auto [tasks, intermediates] = std::move(result);

  TF_ASSERT_OK_AND_ASSIGN(HloSharding output_sharding,
                          GetSharding(kSameOutputSharding));

  EXPECT_THAT(tasks,
              ElementsAre(m::TaskRoots(Each(op::Sharding(output_sharding)))));
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
  custom-call.78 = f32[2,4]{1,0} custom-call(multiply.76, Arg_2.58), custom_call_target="MultiMeshTask", called_computations={f.impl.19}, backend_config={"name": "f", "devices": [0,1], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  Arg_3.59 = f32[4]{0} parameter(3)
  call.79 = f32[] call(custom-call.78, Arg_3.59), to_apply=g.impl.31
  constant.62 = s32[] constant(2)
  add.75 = s32[] add(Arg_4.60, constant.62)
  Arg_5.61 = f32[] parameter(5)
  custom-call.80 = f32[] custom-call(custom-call.78, Arg_3.59), custom_call_target="MultiMeshTask", called_computations={g.impl.45}, backend_config={"name": "g", "devices": [2,3], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
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

TEST_F(MpmdPartitionTest, MultipleMicrobatchSlices) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kMultipleMicrobatchSlicesHlo, /*num_devices=*/2));
  auto [tasks, intermediates] = std::move(result);

  // There should be five tasks, one the loop preamble
  // and next the loop-dependent task that slices x2 microbatches
  // and the final loop-dependent task x2 microbatches
  EXPECT_THAT(tasks, UnorderedElementsAre(Not(m::LoopTask()), m::LoopTask(true),
                                          m::LoopTask(true), m::LoopTask(false),
                                          m::LoopTask(false)));
}

static constexpr absl::string_view kSimpleImplicitTaskOptBarrierHlo = R"(
region_0.10 {
  Arg_0.11 = f32[] parameter(0)
  Arg_1.12 = f32[] parameter(1)
  ROOT add.13 = f32[] add(Arg_0.11, Arg_1.12), metadata={op_name="jit(c)/jit(main)/task_g/reduce_sum[axes=(0,)]"}
}

ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  multiply.4 = f32[8]{0} multiply(Arg_0.1, Arg_0.1), metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  Arg_1.2 = s32[] parameter(1), sharding={replicated}
  convert.5 = f32[] convert(Arg_1.2), metadata={op_name="jit(c)/jit(main)/task_f/convert_element_type[new_dtype=float32 weak_type=False]"}
  broadcast.6 = f32[8]{0} broadcast(convert.5), dimensions={}, metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  multiply.7 = f32[8]{0} multiply(multiply.4, broadcast.6), metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  tuple.8 = (f32[8]{0}, f32[8]{0}) tuple(broadcast.6, multiply.7)
  opt-barrier.9 = (f32[8]{0}, f32[8]{0}) opt-barrier(tuple.8)
  get-tuple-element.10 = f32[8]{0} get-tuple-element(opt-barrier.9), index=0
  get-tuple-element.11 = f32[8]{0} get-tuple-element(opt-barrier.9), index=1
  cosine.8 = f32[8]{0} cosine(get-tuple-element.11), metadata={op_name="jit(c)/jit(main)/task_g/cos"}
  add.9 = f32[8]{0} add(cosine.8, get-tuple-element.11), metadata={op_name="jit(c)/jit(main)/task_g/add"}
  constant.3 = f32[] constant(0)
  ROOT reduce.14 = f32[] reduce(add.9, constant.3), dimensions={0}, to_apply=region_0.10, metadata={op_name="jit(c)/jit(main)/task_g/reduce_sum[axes=(0,)]"}
} // main.15
)";

TEST_F(MpmdPartitionTest, SimpleImplicitTaskOptBarrier) {
  RegisterNamedTestTask("task_f", {0, 2}, {2}, {"x"}, {{"x", "batch"}});
  RegisterNamedTestTask("task_g", {2, 4}, {2}, {"x"}, {{"x", "batch"}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kSimpleImplicitTaskOptBarrierHlo, /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  // the cross-task optimization barrier should have been removed
  EXPECT_THAT(tasks,
              ElementsAre(m::AnyTask(), m::TaskInstructions(Not(Contains(
                                            op::OptimizationBarrier())))));

  EXPECT_THAT(tasks, ElementsAre(m::TaskDevices(ElementsAre(0, 1)),
                                 m::TaskDevices(ElementsAre(2, 3))));
}

static constexpr absl::string_view kMicrobatchPrePostTaskHlo = R"(
region_1.19 {
  Arg_0.20 = s32[] parameter(0)
  Arg_1.21 = s32[] parameter(1)
  ROOT add.22 = s32[] add(Arg_0.20, Arg_1.21)
}

region_0.23 {
  arg_tuple.24 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) parameter(0)
  get-tuple-element.25 = s32[] get-tuple-element(arg_tuple.24), index=0
  constant.31 = s32[] constant(1)
  add.61 = s32[] add(get-tuple-element.25, constant.31)
  get-tuple-element.26 = s32[] get-tuple-element(arg_tuple.24), index=1
  constant.32 = s32[] constant(2)
  add.50 = s32[] add(get-tuple-element.26, constant.32)
  get-tuple-element.27 = s32[4]{0} get-tuple-element(arg_tuple.24), index=2
  get-tuple-element.28 = s32[4,4]{1,0} get-tuple-element(arg_tuple.24), index=3
  constant.34 = s32[] constant(0)
  compare.35 = pred[] compare(get-tuple-element.26, constant.34), direction=LT
  constant.33 = s32[] constant(4)
  add.36 = s32[] add(get-tuple-element.26, constant.33)
  select.37 = s32[] select(compare.35, add.36, get-tuple-element.26)
  dynamic-slice.38 = s32[2,4]{1,0} dynamic-slice(get-tuple-element.28, select.37, constant.34), dynamic_slice_sizes={2,4}
  custom-call.39 = s32[2,4]{1,0} custom-call(dynamic-slice.38), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  get-tuple-element.29 = s32[4,1]{1,0} get-tuple-element(arg_tuple.24), index=4
  compare.40 = pred[] compare(get-tuple-element.26, constant.34), direction=LT
  add.41 = s32[] add(get-tuple-element.26, constant.33)
  select.42 = s32[] select(compare.40, add.41, get-tuple-element.26)
  dynamic-slice.43 = s32[2,1]{1,0} dynamic-slice(get-tuple-element.29, select.42, constant.34), dynamic_slice_sizes={2,1}
  custom-call.44 = s32[2,1]{1,0} custom-call(dynamic-slice.43), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  broadcast.51 = s32[2,1]{1,0} broadcast(custom-call.44), dimensions={0,1}
  reshape.52 = s32[2]{0} reshape(broadcast.51)
  broadcast.53 = s32[2,4]{1,0} broadcast(reshape.52), dimensions={0}
  add.54 = s32[2,4]{1,0} add(custom-call.39, broadcast.53)
  get-tuple-element.30 = s32[4,1]{1,0} get-tuple-element(arg_tuple.24), index=5
  compare.45 = pred[] compare(get-tuple-element.26, constant.34), direction=LT
  add.46 = s32[] add(get-tuple-element.26, constant.33)
  select.47 = s32[] select(compare.45, add.46, get-tuple-element.26)
  dynamic-slice.48 = s32[2,1]{1,0} dynamic-slice(get-tuple-element.30, select.47, constant.34), dynamic_slice_sizes={2,1}
  custom-call.49 = s32[2,1]{1,0} custom-call(dynamic-slice.48), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  broadcast.55 = s32[2,1]{1,0} broadcast(custom-call.49), dimensions={0,1}
  reshape.56 = s32[2]{0} reshape(broadcast.55)
  broadcast.57 = s32[2,4]{1,0} broadcast(reshape.56), dimensions={0}
  add.58 = s32[2,4]{1,0} add(add.54, broadcast.57)
  reduce.59 = s32[4]{0} reduce(add.58, constant.34), dimensions={0}, to_apply=region_1.19
  add.60 = s32[4]{0} add(get-tuple-element.27, reduce.59)
  ROOT tuple.62 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) tuple(add.61, add.50, add.60, get-tuple-element.28, get-tuple-element.29, /*index=5*/get-tuple-element.30)
} // region_0.23

region_2.63 {
  arg_tuple.64 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) parameter(0)
  get-tuple-element.66 = s32[] get-tuple-element(arg_tuple.64), index=1
  get-tuple-element.67 = s32[4]{0} get-tuple-element(arg_tuple.64), index=2
  get-tuple-element.68 = s32[4,4]{1,0} get-tuple-element(arg_tuple.64), index=3
  get-tuple-element.69 = s32[4,1]{1,0} get-tuple-element(arg_tuple.64), index=4
  get-tuple-element.70 = s32[4,1]{1,0} get-tuple-element(arg_tuple.64), index=5
  get-tuple-element.65 = s32[] get-tuple-element(arg_tuple.64), index=0
  constant.71 = s32[] constant(2)
  ROOT compare.72 = pred[] compare(get-tuple-element.65, constant.71), direction=LT
} // region_2.63

region_3.80 {
  Arg_0.81 = s32[] parameter(0)
  Arg_1.82 = s32[] parameter(1)
  ROOT add.83 = s32[] add(Arg_0.81, Arg_1.82)
}

ENTRY main.87 {
  constant.10 = s32[] constant(0)
  constant.4 = s32[] constant(0)
  broadcast.5 = s32[4]{0} broadcast(constant.4), dimensions={}
  custom-call.14 = s32[4]{0} custom-call(broadcast.5), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_0.1 = s32[4,4]{1,0} parameter(0), sharding={replicated}
  constant.8 = s32[] constant(2)
  broadcast.9 = s32[4,4]{1,0} broadcast(constant.8), dimensions={}
  multiply.11 = s32[4,4]{1,0} multiply(Arg_0.1, broadcast.9)
  custom-call.15 = s32[4,4]{1,0} custom-call(multiply.11), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_1.2 = s32[4,1]{1,0} parameter(1), sharding={replicated}
  constant.6 = s32[] constant(2)
  broadcast.7 = s32[4,1]{1,0} broadcast(constant.6), dimensions={}
  multiply.12 = s32[4,1]{1,0} multiply(Arg_1.2, broadcast.7)
  custom-call.16 = s32[4,1]{1,0} custom-call(multiply.12), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_2.3 = s32[4,1]{1,0} parameter(2), sharding={replicated}
  multiply.13 = s32[4,1]{1,0} multiply(Arg_2.3, broadcast.7)
  custom-call.17 = s32[4,1]{1,0} custom-call(multiply.13), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  tuple.18 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) tuple(constant.10, constant.10, custom-call.14, custom-call.15, custom-call.16, /*index=5*/custom-call.17)
  while.73 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) while(tuple.18), condition=region_2.63, body=region_0.23
  get-tuple-element.74 = s32[] get-tuple-element(while.73), index=0
  get-tuple-element.75 = s32[] get-tuple-element(while.73), index=1
  get-tuple-element.77 = s32[4,4]{1,0} get-tuple-element(while.73), index=3
  get-tuple-element.78 = s32[4,1]{1,0} get-tuple-element(while.73), index=4
  get-tuple-element.79 = s32[4,1]{1,0} get-tuple-element(while.73), index=5
  get-tuple-element.76 = s32[4]{0} get-tuple-element(while.73), index=2
  reduce.84 = s32[] reduce(get-tuple-element.76, constant.10), dimensions={0}, to_apply=region_3.80
  tuple.85 = (s32[]) tuple(reduce.84)
  ROOT get-tuple-element.86 = s32[] get-tuple-element(tuple.85), index=0, sharding={replicated}
} // main.87
)";

TEST_F(MpmdPartitionTest, MicrobatchPrePostTask) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kMicrobatchPrePostTaskHlo, /*num_devices=*/1));
  auto [tasks, intermediates] = std::move(result);

  // there should be a preamble task before the microbatching
  // and a loop-dependent task after the microbatching x2 microbatches
  // and a post-loop task
  EXPECT_THAT(tasks, ElementsAre(Not(m::LoopTask()), m::LoopTask(true),
                                 m::LoopTask(true), Not(m::LoopTask())));
}

static constexpr absl::string_view kCallInMicrobatchTaskHlo = R"(
inner_comp.20 {
  Arg_0.21 = f32[2,4]{1,0} parameter(0)
  Arg_1.22 = f32[2,1]{1,0} parameter(1)
  broadcast.23 = f32[2,1]{1,0} broadcast(Arg_1.22), dimensions={0,1}, metadata={op_name="jit(c)/jit(main)/while/body/layer0/jit(inner_comp)/inner/add"}
  reshape.24 = f32[2]{0} reshape(broadcast.23), metadata={op_name="jit(c)/jit(main)/while/body/layer0/jit(inner_comp)/inner/add"}
  broadcast.25 = f32[2,4]{1,0} broadcast(reshape.24), dimensions={0}, metadata={op_name="jit(c)/jit(main)/while/body/layer0/jit(inner_comp)/inner/add"}
  ROOT add.26 = f32[2,4]{1,0} add(Arg_0.21, broadcast.25), metadata={op_name="jit(c)/jit(main)/while/body/layer0/jit(inner_comp)/inner/add"}
} // inner_comp.20

region_1.27 {
  Arg_0.28 = f32[] parameter(0)
  Arg_1.29 = f32[] parameter(1)
  ROOT add.30 = f32[] add(Arg_0.28, Arg_1.29), metadata={op_name="jit(c)/jit(main)/while/body/layer1/reduce_sum[axes=(0,)]"}
}

region_0.31 {
  arg_tuple.32 = (s32[], s32[], f32[4]{0}, f32[4,4]{1,0}, f32[4,1]{1,0}, /*index=5*/f32[4,1]{1,0}) parameter(0)
  get-tuple-element.33 = s32[] get-tuple-element(arg_tuple.32), index=0
  constant.39 = s32[] constant(1)
  add.67 = s32[] add(get-tuple-element.33, constant.39), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  get-tuple-element.34 = s32[] get-tuple-element(arg_tuple.32), index=1
  constant.41 = s32[] constant(2)
  add.59 = s32[] add(get-tuple-element.34, constant.41), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  get-tuple-element.35 = f32[4]{0} get-tuple-element(arg_tuple.32), index=2
  get-tuple-element.36 = f32[4,4]{1,0} get-tuple-element(arg_tuple.32), index=3
  constant.43 = s32[] constant(0)
  compare.44 = pred[] compare(get-tuple-element.34, constant.43), direction=LT, metadata={op_name="jit(c)/jit(main)/while/body/lt"}
  constant.42 = s32[] constant(4)
  add.45 = s32[] add(get-tuple-element.34, constant.42), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  select.46 = s32[] select(compare.44, add.45, get-tuple-element.34), metadata={op_name="jit(c)/jit(main)/while/body/select_n"}
  dynamic-slice.47 = f32[2,4]{1,0} dynamic-slice(get-tuple-element.36, select.46, constant.43), dynamic_slice_sizes={2,4}, metadata={op_name="jit(c)/jit(main)/while/body/dynamic_slice[slice_sizes=(2, 4)]"}
  custom-call.48 = f32[2,4]{1,0} custom-call(dynamic-slice.47), custom_call_target="MicrobatchSlice", metadata={op_name="jit(c)/jit(main)/while/body/MicrobatchSlice/MicrobatchSlice"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  get-tuple-element.37 = f32[4,1]{1,0} get-tuple-element(arg_tuple.32), index=4
  compare.49 = pred[] compare(get-tuple-element.34, constant.43), direction=LT, metadata={op_name="jit(c)/jit(main)/while/body/lt"}
  add.50 = s32[] add(get-tuple-element.34, constant.42), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  select.51 = s32[] select(compare.49, add.50, get-tuple-element.34), metadata={op_name="jit(c)/jit(main)/while/body/select_n"}
  dynamic-slice.52 = f32[2,1]{1,0} dynamic-slice(get-tuple-element.37, select.51, constant.43), dynamic_slice_sizes={2,1}, metadata={op_name="jit(c)/jit(main)/while/body/dynamic_slice[slice_sizes=(2, 1)]"}
  custom-call.53 = f32[2,1]{1,0} custom-call(dynamic-slice.52), custom_call_target="MicrobatchSlice", metadata={op_name="jit(c)/jit(main)/while/body/MicrobatchSlice/MicrobatchSlice"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  call.60 = f32[2,4]{1,0} call(custom-call.48, custom-call.53), to_apply=inner_comp.20
  get-tuple-element.38 = f32[4,1]{1,0} get-tuple-element(arg_tuple.32), index=5
  compare.54 = pred[] compare(get-tuple-element.34, constant.43), direction=LT, metadata={op_name="jit(c)/jit(main)/while/body/lt"}
  add.55 = s32[] add(get-tuple-element.34, constant.42), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  select.56 = s32[] select(compare.54, add.55, get-tuple-element.34), metadata={op_name="jit(c)/jit(main)/while/body/select_n"}
  dynamic-slice.57 = f32[2,1]{1,0} dynamic-slice(get-tuple-element.38, select.56, constant.43), dynamic_slice_sizes={2,1}, metadata={op_name="jit(c)/jit(main)/while/body/dynamic_slice[slice_sizes=(2, 1)]"}
  custom-call.58 = f32[2,1]{1,0} custom-call(dynamic-slice.57), custom_call_target="MicrobatchSlice", metadata={op_name="jit(c)/jit(main)/while/body/MicrobatchSlice/MicrobatchSlice"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  broadcast.61 = f32[2,1]{1,0} broadcast(custom-call.58), dimensions={0,1}, metadata={op_name="jit(c)/jit(main)/while/body/layer1/add"}
  reshape.62 = f32[2]{0} reshape(broadcast.61), metadata={op_name="jit(c)/jit(main)/while/body/layer1/add"}
  broadcast.63 = f32[2,4]{1,0} broadcast(reshape.62), dimensions={0}, metadata={op_name="jit(c)/jit(main)/while/body/layer1/add"}
  add.64 = f32[2,4]{1,0} add(call.60, broadcast.63), metadata={op_name="jit(c)/jit(main)/while/body/layer1/add"}
  constant.40 = f32[] constant(0)
  reduce.65 = f32[4]{0} reduce(add.64, constant.40), dimensions={0}, to_apply=region_1.27, metadata={op_name="jit(c)/jit(main)/while/body/layer1/reduce_sum[axes=(0,)]"}
  add.66 = f32[4]{0} add(get-tuple-element.35, reduce.65), metadata={op_name="jit(c)/jit(main)/while/body/add"}
  ROOT tuple.68 = (s32[], s32[], f32[4]{0}, f32[4,4]{1,0}, f32[4,1]{1,0}, /*index=5*/f32[4,1]{1,0}) tuple(add.67, add.59, add.66, get-tuple-element.36, get-tuple-element.37, /*index=5*/get-tuple-element.38)
} // region_0.31

region_2.69 {
  arg_tuple.70 = (s32[], s32[], f32[4]{0}, f32[4,4]{1,0}, f32[4,1]{1,0}, /*index=5*/f32[4,1]{1,0}) parameter(0)
  get-tuple-element.72 = s32[] get-tuple-element(arg_tuple.70), index=1
  get-tuple-element.73 = f32[4]{0} get-tuple-element(arg_tuple.70), index=2
  get-tuple-element.74 = f32[4,4]{1,0} get-tuple-element(arg_tuple.70), index=3
  get-tuple-element.75 = f32[4,1]{1,0} get-tuple-element(arg_tuple.70), index=4
  get-tuple-element.76 = f32[4,1]{1,0} get-tuple-element(arg_tuple.70), index=5
  get-tuple-element.71 = s32[] get-tuple-element(arg_tuple.70), index=0
  constant.77 = s32[] constant(2)
  ROOT compare.78 = pred[] compare(get-tuple-element.71, constant.77), direction=LT, metadata={op_name="jit(c)/jit(main)/while/cond/lt"}
} // region_2.69

region_3.86 {
  Arg_0.87 = f32[] parameter(0)
  Arg_1.88 = f32[] parameter(1)
  ROOT add.89 = f32[] add(Arg_0.87, Arg_1.88), metadata={op_name="jit(c)/jit(main)/layer1/reduce_sum[axes=(0,)]"}
}

ENTRY main.93 {
  constant.10 = s32[] constant(0)
  constant.4 = f32[] constant(0)
  broadcast.5 = f32[4]{0} broadcast(constant.4), dimensions={}
  custom-call.15 = f32[4]{0} custom-call(broadcast.5), custom_call_target="MicrobatchInit", metadata={op_name="jit(c)/jit(main)/MicrobatchInit/MicrobatchInit"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={replicated}
  constant.8 = f32[] constant(2)
  broadcast.9 = f32[4,4]{1,0} broadcast(constant.8), dimensions={}
  multiply.12 = f32[4,4]{1,0} multiply(Arg_0.1, broadcast.9), metadata={op_name="jit(c)/jit(main)/layer0/mul"}
  custom-call.16 = f32[4,4]{1,0} custom-call(multiply.12), custom_call_target="Microbatch", metadata={op_name="jit(c)/jit(main)/Microbatch/Microbatch"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_1.2 = f32[4,1]{1,0} parameter(1), sharding={replicated}
  constant.6 = f32[] constant(2)
  broadcast.7 = f32[4,1]{1,0} broadcast(constant.6), dimensions={}
  multiply.13 = f32[4,1]{1,0} multiply(Arg_1.2, broadcast.7), metadata={op_name="jit(c)/jit(main)/layer0/mul"}
  custom-call.17 = f32[4,1]{1,0} custom-call(multiply.13), custom_call_target="Microbatch", metadata={op_name="jit(c)/jit(main)/Microbatch/Microbatch"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_2.3 = f32[4,1]{1,0} parameter(2), sharding={replicated}
  multiply.14 = f32[4,1]{1,0} multiply(Arg_2.3, broadcast.7), metadata={op_name="jit(c)/jit(main)/layer0/mul"}
  custom-call.18 = f32[4,1]{1,0} custom-call(multiply.14), custom_call_target="Microbatch", metadata={op_name="jit(c)/jit(main)/Microbatch/Microbatch"}, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  tuple.19 = (s32[], s32[], f32[4]{0}, f32[4,4]{1,0}, f32[4,1]{1,0}, /*index=5*/f32[4,1]{1,0}) tuple(constant.10, constant.10, custom-call.15, custom-call.16, custom-call.17, /*index=5*/custom-call.18), metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  while.79 = (s32[], s32[], f32[4]{0}, f32[4,4]{1,0}, f32[4,1]{1,0}, /*index=5*/f32[4,1]{1,0}) while(tuple.19), condition=region_2.69, body=region_0.31, metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  get-tuple-element.80 = s32[] get-tuple-element(while.79), index=0, metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  get-tuple-element.81 = s32[] get-tuple-element(while.79), index=1, metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  get-tuple-element.83 = f32[4,4]{1,0} get-tuple-element(while.79), index=3, metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  get-tuple-element.84 = f32[4,1]{1,0} get-tuple-element(while.79), index=4, metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  get-tuple-element.85 = f32[4,1]{1,0} get-tuple-element(while.79), index=5, metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  get-tuple-element.82 = f32[4]{0} get-tuple-element(while.79), index=2, metadata={op_name="jit(c)/jit(main)/while[cond_nconsts=0 body_nconsts=3]"}
  constant.11 = f32[] constant(0)
  reduce.90 = f32[] reduce(get-tuple-element.82, constant.11), dimensions={0}, to_apply=region_3.86, metadata={op_name="jit(c)/jit(main)/layer1/reduce_sum[axes=(0,)]"}
  tuple.91 = (f32[]) tuple(reduce.90)
  ROOT get-tuple-element.92 = f32[] get-tuple-element(tuple.91), index=0, sharding={replicated}
} // main.93
)";

TEST_F(MpmdPartitionTest, CallInMicrobatchTask) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kCallInMicrobatchTaskHlo, /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  // there should be a preamble task before the microbatching
  // and a loop-dependent task after the microbatching x2 microbatches
  // and a post-loop task
  EXPECT_THAT(tasks, ElementsAre(Not(m::LoopTask()), m::LoopTask(true),
                                 m::LoopTask(true), Not(m::LoopTask())));
}

static constexpr absl::string_view kDeviceReshardingHlo = R"(
ENTRY main.23 {
  Arg_0.1 = f32[4,2]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  constant.3 = f32[] constant(2)
  broadcast.4 = f32[4,2]{1,0} broadcast(constant.3), dimensions={}
  add.8 = f32[4,2]{1,0} add(Arg_0.1, broadcast.4), sharding={devices=[2,1]0,1}
  Arg_1.2 = f32[4,2]{1,0} parameter(1), sharding={devices=[2,1]2,3}
  cosine.19 = f32[4,2]{1,0} cosine(Arg_1.2)
  add.20 = f32[4,2]{1,0} add(cosine.19, add.8), sharding={devices=[2,1]2,3}
  ROOT multiply.20 = f32[4,2]{1,0} multiply(add.20, add.20), sharding={devices=[2,1]0,1}
} // main.23
)";

TEST_F(MpmdPartitionTest, DeviceResharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, RunMpmdOnHloString(kDeviceReshardingHlo, /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  ASSERT_EQ(tasks.size(), 3);

  EXPECT_THAT(tasks, ElementsAre(m::TaskDevices(ElementsAre(0, 1)),
                                 m::TaskDevices(ElementsAre(2, 3)),
                                 m::TaskDevices(ElementsAre(0, 1))));
}

static constexpr absl::string_view kSupportedIntermediateReshardingHlo = R"(
ENTRY main.23 {
  Arg_0.1 = f32[4,2]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  constant.3 = f32[] constant(2)
  broadcast.4 = f32[4,2]{1,0} broadcast(constant.3), dimensions={}
  custom-call.15 = f32[4,2]{1,0} custom-call(Arg_0.1), custom_call_target="sharding", sharding={devices=[4,1]0,1,2,3}
  multiply.8 = f32[4,2]{1,0} multiply(custom-call.15, broadcast.4), sharding={devices=[4,1]0,1,2,3}
  Arg_1.2 = f32[4,2]{1,0} parameter(1), sharding={devices=[2,1]2,3}
  cosine.19 = f32[4,2]{1,0} cosine(Arg_1.2)
  add.20 = f32[4,2]{1,0} add(cosine.19, multiply.8), sharding={devices=[2,1]2,3}
  ROOT tuple.42 = (f32[4,2]{1,0}, f32[4,2]{1,0}) tuple(add.20, add.20), sharding={{devices=[2,1]0,1},{devices=[2,1]2,3}}
} // main.23
)";

static constexpr absl::string_view kIntermediateOutputSharding = R"(
type: OTHER
tile_assignment_dimensions: 4
tile_assignment_dimensions: 1
iota_reshape_dims: 4
iota_transpose_perm: 0
)";
static constexpr absl::string_view kIntermediateInputSharding = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 1
iota_reshape_dims: 2
iota_transpose_perm: 0
iota_offset: 2
)";

TEST_F(MpmdPartitionTest, SupportedIntermediateResharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, RunMpmdOnHloString(kSupportedIntermediateReshardingHlo,
                                      /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);
  TF_ASSERT_OK_AND_ASSIGN(HloSharding output_sharding,
                          GetSharding(kIntermediateOutputSharding));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input_sharding,
                          GetSharding(kIntermediateInputSharding));

  EXPECT_THAT(
      tasks,
      ElementsAre(Field(&SpmdHloModuleTask::outputs,
                        Contains(m::Store({.index = 1,
                                           .type = Store::Type::TEMP,
                                           .sharding = output_sharding}))),
                  Field(&SpmdHloModuleTask::inputs,
                        Contains(m::Store({.index = 2,
                                           .type = Store::Type::TEMP,
                                           .sharding = input_sharding})))));
}

static constexpr absl::string_view kBarrierAliasHlo = R"(
ENTRY %main.12 {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  Arg_2.3 = f32[4,4]{1,0} parameter(1), sharding={devices=[2,1]0,1}
  reshape.1 = f32[4,4]{1,0} reshape(Arg_0.1)
  add.1 = f32[4,4]{1,0} add(reshape.1,reshape.1), sharding={devices=[4,1]0,1,2,3}, metadata={op_name="layer0"}
  reshape.3 = f32[4,4]{1,0} reshape(Arg_2.3)
  tuple = (f32[4,4],f32[4,4]) tuple(add.1, reshape.3)
  opt-barrier = (f32[4,4],f32[4,4]) opt-barrier(tuple)
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(opt-barrier), index=0
  custom-call.1 = f32[4,4]{1,0} custom-call(get-tuple-element.1), custom_call_target="Sharding", sharding={devices=[2,1]0,1}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(opt-barrier), index=1
  ROOT add.3 = f32[4,4]{1,0} add(custom-call.1, get-tuple-element.3), sharding={devices=[2,1]0,1}, metadata={op_name="layer1"}
}
)";

TEST_F(MpmdPartitionTest, BarrierAlias) {
  // just validate that it gets built
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunMpmdOnHloString(kBarrierAliasHlo,
                                                          /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  EXPECT_THAT(
      tasks,
      ElementsAre(
          AllOf(m::TaskInstructions(Not(Contains(op::OptimizationBarrier()))),
                m::TaskDevices(ElementsAre(0, 1, 2, 3))),
          AllOf(m::TaskInstructions(Not(Contains(op::OptimizationBarrier()))),
                m::TaskDevices(ElementsAre(0, 1)))));
}

static constexpr absl::string_view kReplicatedSubmeshOutputsHlo = R"(
ENTRY %main.4 {
  Arg_0.1 = s32[8]{0} parameter(0), sharding={devices=[2]0,1}
  Arg_1.2 = s32[8]{0} parameter(1), sharding={devices=[2]2,3}
  multiply.3 = s32[8]{0} multiply(Arg_0.1, Arg_0.1), sharding={devices=[2]0,1}
  multiply.4 = s32[8]{0} multiply(Arg_1.2, Arg_1.2), sharding={devices=[2]2,3}
  multiply.5 = s32[8]{0} multiply(multiply.3, Arg_0.1), sharding={replicated}
  multiply.6 = s32[8]{0} multiply(multiply.4, Arg_1.2), sharding={replicated}
  ROOT tuple.2 = (s32[8]{0}, s32[8]{0}, s32[8]{0}, s32[8]{0}) tuple(multiply.3, multiply.5, multiply.4, multiply.6), sharding={{devices=[2]0,1}, {replicated}, {devices=[2]2,3}, {replicated}}
}
)";
static constexpr absl::string_view kReplicatedSubmeshSharding01 = R"(
type: OTHER
tile_assignment_dimensions: 1
tile_assignment_dimensions: 2
replicate_on_last_tile_dim: true
iota_reshape_dims: 2
iota_transpose_perm: 0
)";
static constexpr absl::string_view kReplicatedSubmeshSharding23 = R"(
type: OTHER
tile_assignment_dimensions: 1
tile_assignment_dimensions: 2
replicate_on_last_tile_dim: true
iota_reshape_dims: 2
iota_transpose_perm: 0
iota_offset: 2
)";

TEST_F(MpmdPartitionTest, ReplicatedSubmeshOutputs) {
  // just validate that it gets built
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          RunMpmdOnHloString(kReplicatedSubmeshOutputsHlo,
                                             /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  TF_ASSERT_OK_AND_ASSIGN(HloSharding output_sharding_01,
                          GetSharding(kReplicatedSubmeshSharding01));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding output_sharding_23,
                          GetSharding(kReplicatedSubmeshSharding23));

  EXPECT_THAT(
      tasks,
      UnorderedElementsAre(
          m::TaskOutputs(Contains(m::Store(
              {.type = Store::Type::ROOT, .sharding = output_sharding_01}))),
          m::TaskOutputs(Contains(m::Store(
              {.type = Store::Type::ROOT, .sharding = output_sharding_23})))));
}

static constexpr absl::string_view kArgumentRootHlo = R"(
ENTRY %main.12 {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  Arg_2.3 = f32[4,4]{1,0} parameter(1)
  add.1 = f32[4,4]{1,0} add(Arg_0.1, Arg_0.1), sharding={devices=[2,1]0,1}
  add.3 = f32[4,4]{1,0} add(Arg_2.3, Arg_2.3), sharding={devices=[2,1]2,3}
  ROOT tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.1, add.3, Arg_2.3)
}
)";

TEST_F(MpmdPartitionTest, ArgumentRootHlo) {
  // just validate that it gets built
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunMpmdOnHloString(kArgumentRootHlo,
                                                          /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);
}

static constexpr absl::string_view kSimpleParameterDonationHlo = R"(
HloModule jit_f, input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }
ENTRY %main.12 {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  Arg_2.3 = f32[4,4]{1,0} parameter(1), sharding={devices=[2,1]2,3}
  add.1 = f32[4,4]{1,0} add(Arg_0.1, Arg_0.1), sharding={devices=[2,1]0,1}
  add.3 = f32[4,4]{1,0} add(Arg_2.3, Arg_2.3), sharding={devices=[2,1]2,3}
  ROOT tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.1, add.3, Arg_2.3)
}
)";

TEST_F(MpmdPartitionTest, SimpleParameterDonation) {
  // just validate that it gets built
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          RunMpmdOnHloString(kSimpleParameterDonationHlo,
                                             /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  EXPECT_THAT(
      GetAliasPairs(tasks),
      ElementsAre(ElementsAre(FieldsAre(
                      m::Store({.type = Store::Type::PARAM, .index = 0}),
                      m::Store({.type = Store::Type::ROOT, .index = 0}))),
                  ElementsAre(FieldsAre(
                      // resharded argument
                      m::Store({.type = Store::Type::PARAM, .index = 1}),
                      m::Store({.type = Store::Type::ROOT})))));
}

static constexpr absl::string_view kArgRootDonationHlo = R"(
HloModule jit_f, input_output_alias={ {0}: (0, {}, may-alias), {3}: (1, {}, may-alias) }
ENTRY %main.12 {
  Arg_0.1 = f32[4] parameter(0), sharding={devices=[2]0,1}
  Arg_2.3 = f32[4] parameter(1), sharding={devices=[2]2,3}
  add.1 = f32[4] add(Arg_0.1, Arg_0.1), sharding={devices=[2]0,1}
  add.2 = f32[4] add(add.1, Arg_0.1)
  add.3 = f32[4] add(Arg_2.3, add.2), sharding={devices=[2]2,3}
  ROOT tuple = (f32[4], f32[4], f32[4], f32[4]) tuple(Arg_0.1, add.1, add.3, Arg_2.3)
}
)";

TEST_F(MpmdPartitionTest, ArgRootDonation) {
  // just validate that it gets built
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunMpmdOnHloString(kArgRootDonationHlo,
                                                          /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  EXPECT_THAT(
      GetAliasPairs(tasks),
      ElementsAre(Contains(FieldsAre(
                      m::Store({.type = Store::Type::PARAM, .index = 0}),
                      m::Store({.type = Store::Type::ROOT, .index = 0}))),
                  Contains(FieldsAre(
                      m::Store({.type = Store::Type::PARAM, .index = 1}),
                      m::Store({.type = Store::Type::ROOT, .index = 3})))));
}

static constexpr absl::string_view kBufferDonorHlo = R"(
HloModule jit_f, buffer_donor={ (0, {}), (2, {}) }
ENTRY %main.12 {
  Arg_0.1 = f32[4] parameter(0), sharding={devices=[2]0,1}
  Arg_1.2 = f32[4] parameter(1), sharding={devices=[2]0,1}
  Arg_2.3 = f32[4] parameter(2), sharding={devices=[2]2,3}
  add.1 = f32[4] add(Arg_0.1, Arg_1.2), sharding={devices=[2]0,1}
  add.2 = f32[4] add(add.1, Arg_0.1)
  add.3 = f32[4] add(Arg_2.3, add.2), sharding={devices=[2]2,3}
  add.4 = f32[4] add(add.3, Arg_0.1), sharding={devices=[2]0,1}
  add.5 = f32[4] add(add.4, Arg_2.3), sharding={devices=[2]2,3}
  ROOT tuple = (f32[4], f32[4], f32[4], f32[4]) tuple(Arg_0.1, add.4, add.5, Arg_2.3)
}
)";

TEST_F(MpmdPartitionTest, BufferDonor) {
  // just validate that it gets built
  TF_ASSERT_OK_AND_ASSIGN(auto result, RunMpmdOnHloString(kBufferDonorHlo,
                                                          /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  EXPECT_THAT(
      GetAliasPairs(tasks),
      IsSupersetOf({ElementsAre(FieldsAre(
                        m::Store({.type = Store::Type::PARAM, .index = 0}),
                        m::Store({.type = Store::Type::ROOT, .index = 1}))),
                    ElementsAre(FieldsAre(
                        m::Store({.type = Store::Type::PARAM, .index = 2}),
                        m::Store({.type = Store::Type::ROOT})))}));

  EXPECT_THAT(GetAliasPairs(tasks), Contains(IsEmpty()).Times(2));
}

static constexpr absl::string_view kBufferDonorBestPairingHlo = R"(
HloModule jit_f, buffer_donor={ (0, {}), (2, {}) , (3, {})}
ENTRY %main.12 {
  Arg_0.1 = f32[4] parameter(0), sharding={devices=[2]0,1}
  Arg_1.2 = f32[4] parameter(1), sharding={devices=[2]0,1}
  Arg_3.4 = f32[4] parameter(3), sharding={devices=[2]0,1}
  Arg_2.3 = f32[4] parameter(2), sharding={devices=[2]2,3}
  add.1 = f32[4] add(Arg_0.1, Arg_1.2), sharding={devices=[2]0,1}
  add.2 = f32[4] add(add.1, Arg_0.1)
  add.4 = f32[4] add(add.2, Arg_0.1), sharding={devices=[2]0,1}
  add.3 = f32[4] add(Arg_2.3, add.2), sharding={devices=[2]2,3}
  add.5 = f32[4] add(add.2, Arg_2.3), sharding={devices=[2]2,3}
  constant = f32[] constant(0)
  broadcast.4 = f32[4] broadcast(constant), dimensions={}
  multiply.4 = f32[4] multiply(broadcast.4, Arg_3.4), sharding={devices=[2]0,1}
  broadcast.5 = f32[4] broadcast(constant), dimensions={}
  multiply.5 = f32[4] multiply(broadcast.5, broadcast.5), sharding={devices=[2]2,3}
  ROOT tuple = (f32[4], f32[4], f32[4], f32[4]) tuple(add.4, multiply.4, add.5, multiply.5)
}
)";

TEST_F(MpmdPartitionTest, BufferDonorBestPairing) {
  // just validate that it gets built
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          RunMpmdOnHloString(kBufferDonorBestPairingHlo,
                                             /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  EXPECT_THAT(
      GetAliasPairs(tasks),
      UnorderedElementsAre(
          UnorderedElementsAre(
              FieldsAre(m::Store({.type = Store::Type::PARAM, .index = 0}),
                        m::Store({.type = Store::Type::ROOT, .index = 0})),
              FieldsAre(m::Store({.type = Store::Type::PARAM, .index = 3}),
                        m::Store({.type = Store::Type::ROOT, .index = 1}))),
          ElementsAre(
              FieldsAre(m::Store({.type = Store::Type::PARAM, .index = 2}),
                        m::Store({.type = Store::Type::ROOT, .index = 2})))));
}

constexpr absl::string_view kTaskAutoShardingHlo = R"(
HloModule jit_c, entry_computation_layout={(s32[4,4]{1,0}, s32[4,4]{1,0})->s32[4,4]{1,0}}, num_partitions=8

f.impl.12 {
  Arg_0.1 = s32[4,4]{1,0} parameter(0)
  ROOT multiply.6 = s32[4,4]{1,0} multiply(Arg_0.1, Arg_0.1)
} // f.impl.12

g.impl.12 {
  Arg_0.1 = s32[4,4]{1,0} parameter(0)
  Arg_2.3 = s32[4,4]{1,0} parameter(1)
  ROOT add.16 = s32[4,4]{1,0} add(Arg_0.1, Arg_2.3)
} // g.impl.12

ENTRY main.19 {
  Arg_0.1 = s32[4,4]{1,0} parameter(0)
  custom-call.5 = s32[4,4]{1,0} custom-call(Arg_0.1), custom_call_target="AutoSharding", backend_config={"axes": [["batch"], ["model"]]}
  Arg_1.2 = s32[4,4]{1,0} parameter(1)
  custom-call.12 = s32[4,4]{1,0} custom-call(Arg_1.2), custom_call_target="AutoSharding", backend_config={"axes": [["batch"], ["model"]]}
  custom-call.19 = s32[4,4]{1,0} custom-call(custom-call.5), custom_call_target="MultiMeshTask", called_computations={f.impl.12}, metadata={op_name="jit(c)/jit(main)/mm_task"}, backend_config={"name": "f", "devices": [0,1,2,3], "autosharding": {"dims": [2, 2], "device_axes": ["x", "y"], "logical_axes": [["batch", "x"], ["model", "y"]]}}
  ROOT custom-call.20 = s32[4,4]{1,0} custom-call(custom-call.19, custom-call.12), custom_call_target="MultiMeshTask", called_computations={g.impl.12}, metadata={op_name="jit(c)/jit(main)/mm_task"}, backend_config={"name": "g", "devices": [4,5,6,7], "autosharding": {"dims": [2, 2], "device_axes": ["x", "y"], "logical_axes": [["batch", "x"], ["model", "y"]]}}
} // main.19
)";

static constexpr absl::string_view kTaskAutoShardingTask0Pbtxt = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 2
iota_reshape_dims: 4
iota_transpose_perm: 0
)";

static constexpr absl::string_view kTaskAutoShardingTask1Pbtxt = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 2
iota_reshape_dims: 4
iota_transpose_perm: 0
iota_offset: 4
)";

TEST_F(MpmdPartitionTest, TaskAutoSharding) {
  // just validate that it gets built
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kTaskAutoShardingHlo,
                         /*num_devices=*/8, {.use_auto_input_sharding = true}));
  auto [tasks, intermediates] = std::move(result);

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding0,
                          GetSharding(kTaskAutoShardingTask0Pbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding1,
                          GetSharding(kTaskAutoShardingTask1Pbtxt));

  EXPECT_THAT(
      tasks,
      ElementsAre(
          AllOf(m::TaskDevices(ElementsAre(0, 1, 2, 3)),
                m::TaskInputs(ElementsAre(m::Store({.sharding = sharding0}))),
                m::TaskOutputs(ElementsAre(m::Store({.sharding = sharding0})))),
          AllOf(
              m::TaskDevices(ElementsAre(4, 5, 6, 7)),
              m::TaskInputs(Each(m::Store({.sharding = sharding1}))),
              m::TaskOutputs(ElementsAre(m::Store({.sharding = sharding1}))))));
}

static constexpr absl::string_view kMpmdMicrobatchHlo = R"(
HloModule jit_wrapped, entry_computation_layout={(s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0})->(s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}

region_1.14 {
  Arg_0.15 = s32[] parameter(0)
  Arg_1.16 = s32[] parameter(1)
  ROOT add.17 = s32[] add(Arg_0.15, Arg_1.16)
}

region_0.18 {
  arg_tuple.19 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  get-tuple-element.20 = s32[] get-tuple-element(arg_tuple.19), index=0
  constant.30 = s32[] constant(1)
  add.54 = s32[] add(get-tuple-element.20, constant.30)
  get-tuple-element.21 = s32[] get-tuple-element(arg_tuple.19), index=1
  constant.31 = s32[] constant(2)
  add.39 = s32[] add(get-tuple-element.21, constant.31)
  get-tuple-element.22 = s32[] get-tuple-element(arg_tuple.19), index=2
  get-tuple-element.26 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=6
  constant.33 = s32[] constant(0)
  compare.34 = pred[] compare(get-tuple-element.21, constant.33), direction=LT
  constant.32 = s32[] constant(4)
  add.35 = s32[] add(get-tuple-element.21, constant.32)
  select.36 = s32[] select(compare.34, add.35, get-tuple-element.21)
  dynamic-slice.37 = s32[2,4]{1,0} dynamic-slice(get-tuple-element.26, select.36, constant.33), dynamic_slice_sizes={2,4}
  custom-call.38 = s32[2,4]{1,0} custom-call(dynamic-slice.37), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  get-tuple-element.27 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=7
  dot.40 = s32[2,4]{1,0} dot(custom-call.38, get-tuple-element.27), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer0"}
  get-tuple-element.28 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=8
  dot.41 = s32[2,4]{1,0} dot(dot.40, get-tuple-element.28), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer1"}
  get-tuple-element.29 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=9
  dot.42 = s32[2,4]{1,0} dot(dot.41, get-tuple-element.29), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer2"}
  reduce.43 = s32[] reduce(dot.42, constant.33), dimensions={0,1}, to_apply=region_1.14
  add.50 = s32[] add(get-tuple-element.22, reduce.43)
  get-tuple-element.23 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=3
  broadcast.44 = s32[4,4]{1,0} broadcast(reduce.43), dimensions={}
  multiply.45 = s32[4,4]{1,0} multiply(get-tuple-element.27, broadcast.44)
  add.51 = s32[4,4]{1,0} add(get-tuple-element.23, multiply.45)
  get-tuple-element.24 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=4
  broadcast.46 = s32[4,4]{1,0} broadcast(reduce.43), dimensions={}
  multiply.47 = s32[4,4]{1,0} multiply(get-tuple-element.28, broadcast.46)
  add.52 = s32[4,4]{1,0} add(get-tuple-element.24, multiply.47)
  get-tuple-element.25 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=5
  broadcast.48 = s32[4,4]{1,0} broadcast(reduce.43), dimensions={}
  multiply.49 = s32[4,4]{1,0} multiply(get-tuple-element.29, broadcast.48)
  add.53 = s32[4,4]{1,0} add(get-tuple-element.25, multiply.49)
  ROOT tuple.55 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(add.54, add.39, add.50, add.51, add.52, /*index=5*/add.53, get-tuple-element.26, get-tuple-element.27, get-tuple-element.28, get-tuple-element.29)
} // region_0.18

region_2.56 {
  arg_tuple.57 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  get-tuple-element.59 = s32[] get-tuple-element(arg_tuple.57), index=1
  get-tuple-element.60 = s32[] get-tuple-element(arg_tuple.57), index=2
  get-tuple-element.61 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=3
  get-tuple-element.62 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=4
  get-tuple-element.63 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=5
  get-tuple-element.64 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=6
  get-tuple-element.65 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=7
  get-tuple-element.66 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=8
  get-tuple-element.67 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=9
  get-tuple-element.58 = s32[] get-tuple-element(arg_tuple.57), index=0
  constant.68 = s32[] constant(2)
  ROOT compare.69 = pred[] compare(get-tuple-element.58, constant.68), direction=LT
} // region_2.56

ENTRY main.82 {
  constant.5 = s32[] constant(0)
  custom-call.8 = s32[] custom-call(constant.5), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  constant.6 = s32[] constant(0)
  broadcast.7 = s32[4,4]{1,0} broadcast(constant.6), dimensions={}
  custom-call.9 = s32[4,4]{1,0} custom-call(broadcast.7), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  custom-call.10 = s32[4,4]{1,0} custom-call(broadcast.7), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  custom-call.11 = s32[4,4]{1,0} custom-call(broadcast.7), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_0.1 = s32[4,4]{1,0} parameter(0), sharding={replicated}
  custom-call.12 = s32[4,4]{1,0} custom-call(Arg_0.1), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_1.2 = s32[4,4]{1,0} parameter(1), sharding={replicated}
  Arg_2.3 = s32[4,4]{1,0} parameter(2), sharding={replicated}
  Arg_3.4 = s32[4,4]{1,0} parameter(3), sharding={replicated}
  tuple.13 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(constant.5, constant.5, custom-call.8, custom-call.9, custom-call.10, /*index=5*/custom-call.11, custom-call.12, Arg_1.2, Arg_2.3, Arg_3.4)
  while.70 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) while(tuple.13), condition=region_2.56, body=region_0.18
  get-tuple-element.71 = s32[] get-tuple-element(while.70), index=0
  get-tuple-element.72 = s32[] get-tuple-element(while.70), index=1
  get-tuple-element.77 = s32[4,4]{1,0} get-tuple-element(while.70), index=6
  get-tuple-element.78 = s32[4,4]{1,0} get-tuple-element(while.70), index=7
  get-tuple-element.79 = s32[4,4]{1,0} get-tuple-element(while.70), index=8
  get-tuple-element.80 = s32[4,4]{1,0} get-tuple-element(while.70), index=9
  get-tuple-element.73 = s32[] get-tuple-element(while.70), index=2
  get-tuple-element.74 = s32[4,4]{1,0} get-tuple-element(while.70), index=3
  get-tuple-element.75 = s32[4,4]{1,0} get-tuple-element(while.70), index=4
  get-tuple-element.76 = s32[4,4]{1,0} get-tuple-element(while.70), index=5
  ROOT tuple.81 = (s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(get-tuple-element.73, get-tuple-element.74, get-tuple-element.75, get-tuple-element.76)
}
)";

TEST_F(MpmdPartitionTest, MpmdMicrobatch) {
  RegisterNamedTestTask("layer0", {0, 4}, {4}, {"x"}, {{"x", "batch"}});
  RegisterNamedTestTask("layer1", {0, 2}, {2}, {"x"}, {{"x", "batch"}});
  RegisterNamedTestTask("layer2", {0, 4}, {4}, {"x"}, {{"x", "batch"}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kMpmdMicrobatchHlo,
                         /*num_devices=*/4, {.use_auto_input_sharding = true}));
  auto [tasks, intermediates] = std::move(result);

  EXPECT_THAT(tasks,
              AllOf(Contains(AllOf(m::TaskDevices(ElementsAre(0, 1, 2, 3)),
                                   Not(m::LoopTask()))),
                    Contains(AllOf(m::TaskDevices(ElementsAre(0, 1, 2, 3)),
                                   m::LoopTask(true)))
                        .Times(2),
                    Contains(AllOf(m::TaskDevices(ElementsAre(0, 1)),
                                   m::LoopTask(false)))
                        .Times(2),
                    Contains(AllOf(m::TaskDevices(ElementsAre(0, 1, 2, 3)),
                                   m::LoopTask(false)))
                        .Times(2)));
}

static constexpr absl::string_view kMicrobatchWhileBufferDonorHlo = R"(
HloModule jit_f, buffer_donor={ (1, {}), (2, {}), (3, {}) }
region_1.14 {
  Arg_0.15 = s32[] parameter(0)
  Arg_1.16 = s32[] parameter(1)
  ROOT add.17 = s32[] add(Arg_0.15, Arg_1.16)
}

region_0.18 {
  arg_tuple.19 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  get-tuple-element.20 = s32[] get-tuple-element(arg_tuple.19), index=0
  constant.30 = s32[] constant(1)
  add.54 = s32[] add(get-tuple-element.20, constant.30)
  get-tuple-element.21 = s32[] get-tuple-element(arg_tuple.19), index=1
  constant.31 = s32[] constant(2)
  add.39 = s32[] add(get-tuple-element.21, constant.31)
  get-tuple-element.22 = s32[] get-tuple-element(arg_tuple.19), index=2
  get-tuple-element.26 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=6
  constant.33 = s32[] constant(0)
  compare.34 = pred[] compare(get-tuple-element.21, constant.33), direction=LT
  constant.32 = s32[] constant(4)
  add.35 = s32[] add(get-tuple-element.21, constant.32)
  select.36 = s32[] select(compare.34, add.35, get-tuple-element.21)
  dynamic-slice.37 = s32[2,4]{1,0} dynamic-slice(get-tuple-element.26, select.36, constant.33), dynamic_slice_sizes={2,4}
  custom-call.38 = s32[2,4]{1,0} custom-call(dynamic-slice.37), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  get-tuple-element.27 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=7
  dot.40 = s32[2,4]{1,0} dot(custom-call.38, get-tuple-element.27), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer0"}
  get-tuple-element.28 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=8
  dot.41 = s32[2,4]{1,0} dot(dot.40, get-tuple-element.28), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer1"}
  get-tuple-element.29 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=9
  dot.42 = s32[2,4]{1,0} dot(dot.41, get-tuple-element.29), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer2"}
  reduce.43 = s32[] reduce(dot.42, constant.33), dimensions={0,1}, to_apply=region_1.14
  add.50 = s32[] add(get-tuple-element.22, reduce.43)
  get-tuple-element.23 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=3
  broadcast.44 = s32[4,4]{1,0} broadcast(reduce.43), dimensions={}
  multiply.45 = s32[4,4]{1,0} multiply(get-tuple-element.27, broadcast.44)
  add.51 = s32[4,4]{1,0} add(get-tuple-element.23, multiply.45)
  get-tuple-element.24 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=4
  broadcast.46 = s32[4,4]{1,0} broadcast(reduce.43), dimensions={}
  multiply.47 = s32[4,4]{1,0} multiply(get-tuple-element.28, broadcast.46)
  add.52 = s32[4,4]{1,0} add(get-tuple-element.24, multiply.47)
  get-tuple-element.25 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=5
  broadcast.48 = s32[4,4]{1,0} broadcast(reduce.43), dimensions={}
  multiply.49 = s32[4,4]{1,0} multiply(get-tuple-element.29, broadcast.48)
  add.53 = s32[4,4]{1,0} add(get-tuple-element.25, multiply.49)
  ROOT tuple.55 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(add.54, add.39, add.50, add.51, add.52, /*index=5*/add.53, get-tuple-element.26, get-tuple-element.27, get-tuple-element.28, get-tuple-element.29)
} // region_0.18

region_2.56 {
  arg_tuple.57 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  get-tuple-element.59 = s32[] get-tuple-element(arg_tuple.57), index=1
  get-tuple-element.60 = s32[] get-tuple-element(arg_tuple.57), index=2
  get-tuple-element.61 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=3
  get-tuple-element.62 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=4
  get-tuple-element.63 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=5
  get-tuple-element.64 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=6
  get-tuple-element.65 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=7
  get-tuple-element.66 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=8
  get-tuple-element.67 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=9
  get-tuple-element.58 = s32[] get-tuple-element(arg_tuple.57), index=0
  constant.68 = s32[] constant(2)
  ROOT compare.69 = pred[] compare(get-tuple-element.58, constant.68), direction=LT
} // region_2.56

ENTRY main.82 {
  constant.5 = s32[] constant(0)
  custom-call.8 = s32[] custom-call(constant.5), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  constant.6 = s32[] constant(0)
  broadcast.7 = s32[4,4]{1,0} broadcast(constant.6), dimensions={}
  custom-call.9 = s32[4,4]{1,0} custom-call(broadcast.7), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  custom-call.10 = s32[4,4]{1,0} custom-call(broadcast.7), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  custom-call.11 = s32[4,4]{1,0} custom-call(broadcast.7), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_0.1 = s32[4,4]{1,0} parameter(0), sharding={replicated}
  custom-call.12 = s32[4,4]{1,0} custom-call(Arg_0.1), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_1.2 = s32[4,4]{1,0} parameter(1), sharding={replicated}
  Arg_2.3 = s32[4,4]{1,0} parameter(2), sharding={replicated}
  Arg_3.4 = s32[4,4]{1,0} parameter(3), sharding={replicated}
  tuple.13 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(constant.5, constant.5, custom-call.8, custom-call.9, custom-call.10, /*index=5*/custom-call.11, custom-call.12, Arg_1.2, Arg_2.3, Arg_3.4)
  while.70 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) while(tuple.13), condition=region_2.56, body=region_0.18
  get-tuple-element.71 = s32[] get-tuple-element(while.70), index=0
  get-tuple-element.72 = s32[] get-tuple-element(while.70), index=1
  get-tuple-element.77 = s32[4,4]{1,0} get-tuple-element(while.70), index=6
  get-tuple-element.78 = s32[4,4]{1,0} get-tuple-element(while.70), index=7
  get-tuple-element.79 = s32[4,4]{1,0} get-tuple-element(while.70), index=8
  get-tuple-element.80 = s32[4,4]{1,0} get-tuple-element(while.70), index=9
  get-tuple-element.73 = s32[] get-tuple-element(while.70), index=2
  get-tuple-element.74 = s32[4,4]{1,0} get-tuple-element(while.70), index=3
  get-tuple-element.75 = s32[4,4]{1,0} get-tuple-element(while.70), index=4
  get-tuple-element.76 = s32[4,4]{1,0} get-tuple-element(while.70), index=5
  add.77 = s32[4,4]{1,0} add(Arg_1.2, get-tuple-element.74)
  add.78 = s32[4,4]{1,0} add(Arg_2.3, get-tuple-element.75)
  add.79 = s32[4,4]{1,0} add(Arg_3.4, get-tuple-element.76)
  ROOT root_tuple = (s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(add.77, add.78, add.79)
}
)";

TEST_F(MpmdPartitionTest, MicrobatchBufferDonor) {
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          RunMpmdOnHloString(kMicrobatchWhileBufferDonorHlo,
                                             /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);

  EXPECT_THAT(
      GetAliasPairs(tasks),
      ElementsAre(
          IsEmpty(),
          // the middle task has 3-loop carried temps x2 microbatches
          UnorderedElementsAre(
              FieldsAre(m::Store({.type = Store::Type::TEMP}),
                        m::Store({.type = Store::Type::TEMP})),
              FieldsAre(m::Store({.type = Store::Type::TEMP}),
                        m::Store({.type = Store::Type::TEMP})),
              FieldsAre(m::Store({.type = Store::Type::TEMP}),
                        m::Store({.type = Store::Type::TEMP}))),
          UnorderedElementsAre(
              FieldsAre(m::Store({.type = Store::Type::TEMP}),
                        m::Store({.type = Store::Type::TEMP})),
              FieldsAre(m::Store({.type = Store::Type::TEMP}),
                        m::Store({.type = Store::Type::TEMP})),
              FieldsAre(m::Store({.type = Store::Type::TEMP}),
                        m::Store({.type = Store::Type::TEMP}))),
          // the final task sums into the the aliased params
          UnorderedElementsAre(
              FieldsAre(m::Store({.type = Store::Type::PARAM, .index = 1}),
                        m::Store({.type = Store::Type::ROOT})),
              FieldsAre(m::Store({.type = Store::Type::PARAM, .index = 2}),
                        m::Store({.type = Store::Type::ROOT})),
              FieldsAre(m::Store({.type = Store::Type::PARAM, .index = 3}),
                        m::Store({.type = Store::Type::ROOT})))));
}

static constexpr absl::string_view kLoopCarriedShardingHlo = R"(
HloModule jit_wrapped

region_1.10 {
  Arg_0.11 = s32[] parameter(0)
  Arg_1.12 = s32[] parameter(1)
  ROOT add.13 = s32[] add(Arg_0.11, Arg_1.12)
}

region_0.14 {
  arg_tuple.15 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}) parameter(0)
  get-tuple-element.16 = s32[] get-tuple-element(arg_tuple.15), index=0
  constant.22 = s32[] constant(1)
  add.38 = s32[] add(get-tuple-element.16, constant.22)
  get-tuple-element.17 = s32[] get-tuple-element(arg_tuple.15), index=1
  constant.23 = s32[] constant(2)
  add.31 = s32[] add(get-tuple-element.17, constant.23)
  get-tuple-element.18 = s32[] get-tuple-element(arg_tuple.15), index=2
  get-tuple-element.20 = s32[4,4]{1,0} get-tuple-element(arg_tuple.15), index=4
  constant.25 = s32[] constant(0)
  compare.26 = pred[] compare(get-tuple-element.17, constant.25), direction=LT
  constant.24 = s32[] constant(4)
  add.27 = s32[] add(get-tuple-element.17, constant.24)
  select.28 = s32[] select(compare.26, add.27, get-tuple-element.17)
  dynamic-slice.29 = s32[2,4]{1,0} dynamic-slice(get-tuple-element.20, select.28, constant.25), dynamic_slice_sizes={2,4}
  custom-call.30 = s32[2,4]{1,0} custom-call(dynamic-slice.29), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  get-tuple-element.21 = s32[4,4]{1,0} get-tuple-element(arg_tuple.15), index=5
  dot.32 = s32[2,4]{1,0} dot(custom-call.30, get-tuple-element.21), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[1,4]0,1,2,3}
  reduce.33 = s32[] reduce(dot.32, constant.25), dimensions={0,1}, to_apply=region_1.10
  add.36 = s32[] add(get-tuple-element.18, reduce.33)
  get-tuple-element.19 = s32[4,4]{1,0} get-tuple-element(arg_tuple.15), index=3
  broadcast.34 = s32[4,4]{1,0} broadcast(reduce.33), dimensions={}
  multiply.35 = s32[4,4]{1,0} multiply(get-tuple-element.21, broadcast.34), sharding={devices=[1,4]0,1,2,3}
  add.37 = s32[4,4]{1,0} add(get-tuple-element.19, multiply.35)
  ROOT tuple.39 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}) tuple(add.38, add.31, add.36, add.37, get-tuple-element.20, /*index=5*/get-tuple-element.21)
} // region_0.14

region_2.40 {
  arg_tuple.41 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}) parameter(0)
  get-tuple-element.43 = s32[] get-tuple-element(arg_tuple.41), index=1
  get-tuple-element.44 = s32[] get-tuple-element(arg_tuple.41), index=2
  get-tuple-element.45 = s32[4,4]{1,0} get-tuple-element(arg_tuple.41), index=3
  get-tuple-element.46 = s32[4,4]{1,0} get-tuple-element(arg_tuple.41), index=4
  get-tuple-element.47 = s32[4,4]{1,0} get-tuple-element(arg_tuple.41), index=5
  get-tuple-element.42 = s32[] get-tuple-element(arg_tuple.41), index=0
  constant.48 = s32[] constant(2)
  ROOT compare.49 = pred[] compare(get-tuple-element.42, constant.48), direction=LT
} // region_2.40

ENTRY main.58 {
  constant.3 = s32[] constant(0)
  custom-call.6 = s32[] custom-call(constant.3), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  constant.4 = s32[] constant(0)
  broadcast.5 = s32[4,4]{1,0} broadcast(constant.4), dimensions={}
  custom-call.7 = s32[4,4]{1,0} custom-call(broadcast.5), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_0.1 = s32[4,4]{1,0} parameter(0), sharding={replicated}
  custom-call.8 = s32[4,4]{1,0} custom-call(Arg_0.1), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_1.2 = s32[4,4]{1,0} parameter(1), sharding={replicated}
  tuple.9 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}) tuple(constant.3, constant.3, custom-call.6, custom-call.7, custom-call.8, /*index=5*/Arg_1.2)
  while.50 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}) while(tuple.9), condition=region_2.40, body=region_0.14
  get-tuple-element.51 = s32[] get-tuple-element(while.50), index=0
  get-tuple-element.52 = s32[] get-tuple-element(while.50), index=1
  get-tuple-element.55 = s32[4,4]{1,0} get-tuple-element(while.50), index=4
  get-tuple-element.56 = s32[4,4]{1,0} get-tuple-element(while.50), index=5
  get-tuple-element.53 = s32[] get-tuple-element(while.50), index=2
  get-tuple-element.54 = s32[4,4]{1,0} get-tuple-element(while.50), index=3
  ROOT tuple.57 = (s32[], s32[4,4]{1,0}) tuple(get-tuple-element.53, get-tuple-element.54)
} // main.58
)";

static constexpr absl::string_view kLoopCarriedShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 1
tile_assignment_dimensions: 4
iota_reshape_dims: 4
iota_transpose_perm: 0
)";

TEST_F(MpmdPartitionTest, LoopCarriedSharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kLoopCarriedShardingHlo,
                         /*num_devices=*/4,
                         {.use_module_config_auto_param_sharding = true}));
  auto [tasks, intermediates] = std::move(result);

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding,
                          GetSharding(kLoopCarriedShardingPbtxt));

  HloSharding replicated = HloSharding::Replicate();

  EXPECT_THAT(tasks, ElementsAre(Not(m::LoopTask()), m::LoopTask(true),
                                 m::LoopTask(true)));

  EXPECT_THAT(
      tasks,
      ElementsAre(
          m::TaskOutputs(IsSupersetOf(
              {m::Store({.type = Store::Type::ROOT, .sharding = replicated}),
               m::Store({.type = Store::Type::ROOT, .sharding = sharding})})),
          AllOf(  // microbatch zero
              m::TaskInputs(UnorderedElementsAre(
                  m::Store(
                      {.type = Store::Type::PARAM, .sharding = replicated}),
                  m::Store(
                      {.type = Store::Type::PARAM, .sharding = replicated}),
                  m::Store({.type = Store::Type::TEMP,
                            .sharding = replicated}),  // slice
                  m::Store({.type = Store::Type::ROOT,
                            .sharding = sharding}),  // loop-carried
                  m::Store(
                      {.type = Store::Type::ROOT, .sharding = replicated})))),
          AllOf(  // microbatch 1
              m::TaskInputs(UnorderedElementsAre(
                  m::Store(
                      {.type = Store::Type::PARAM, .sharding = replicated}),
                  m::Store(
                      {.type = Store::Type::PARAM, .sharding = replicated}),
                  m::Store({.type = Store::Type::TEMP,
                            .sharding = replicated}),  // slice
                  m::Store({.type = Store::Type::ROOT,
                            .sharding = sharding}),  // loop-carried
                  m::Store(
                      {.type = Store::Type::ROOT, .sharding = replicated}))),
              m::TaskOutputs(UnorderedElementsAre(
                  m::Store({.type = Store::Type::ROOT, .sharding = replicated}),
                  m::Store(
                      {.type = Store::Type::ROOT, .sharding = sharding})))))

  );
}

static constexpr absl::string_view kConcatenateReplicateHlo = R"(
region {
  Arg_0.10 = f32[] parameter(0)
  Arg_1.11 = f32[] parameter(1)
  ROOT add.12 = f32[] add(Arg_0.10, Arg_1.11)
}

ENTRY %main.12 {
  Arg_0.1 = f32[4,4]{1,0} parameter(0)
  Arg_2.3 = f32[4,4]{1,0} parameter(1)
  add.1 = f32[4,4]{1,0} add(Arg_0.1, Arg_0.1), sharding={devices=[2,1]0,1}
  add.2 = f32[4,4]{1,0} add(Arg_2.3, Arg_2.3), sharding={devices=[2,1]2,3}
  constant = f32[] constant(0)
  reshape.1 = f32[16]{0} reshape(add.1)
  reduce.1 = f32[] reduce(reshape.1, constant), dimensions={0}, to_apply=region
  reduce.1.clone = f32[] reduce(reshape.1, constant), dimensions={0}, to_apply=region
  reshape.2 = f32[16]{0} reshape(add.2)
  reduce.2 = f32[] reduce(reshape.2, constant), dimensions={0}, to_apply=region
  reduce.2.clone = f32[] reduce(reshape.2, constant), dimensions={0}, to_apply=region
  reshape.11 = f32[1]{0} reshape(reduce.1)
  reshape.22 = f32[1]{0} reshape(reduce.2)
  reshape.11.clone = f32[1]{0} reshape(reduce.1.clone)
  reshape.22.clone = f32[1]{0} reshape(reduce.2.clone)
  concatenate = f32[4] concatenate(reshape.11, reshape.22, reshape.11.clone, reshape.22.clone), dimensions={0}
  reduce.3 = f32[] reduce(concatenate, constant), dimensions={0}, to_apply=region
  sqrt = f32[] sqrt(reduce.3)
  broadcast = f32[4,4]{1,0} broadcast(sqrt), dimensions={}
  multiply.1 = f32[4,4]{1,0} multiply(broadcast, Arg_0.1)
  multiply.2 = f32[4,4]{1,0} multiply(broadcast, Arg_2.3)
  ROOT tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(multiply.1, multiply.2)
}
)";

TEST_F(MpmdPartitionTest, ConcatenateReplicate) {
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          RunMpmdOnHloString(kConcatenateReplicateHlo,
                                             /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);
  EXPECT_THAT(
      tasks,
      IsSupersetOf(
          {// the first task should concatante its two operands
           m::TaskInstructions(
               Contains(op::Concatenate(op::Reshape(), op::Reshape()))
                   .Times(1)),
           // the second task should concatenate its two operands
           // and then do the aggregate concatenate
           m::TaskInstructions(AllOf(
               Contains(op::Concatenate(op::Reshape(), op::Reshape())).Times(1),
               Contains(op::Concatenate(op::Parameter(), op::Concatenate()))
                   .Times(1)))}));
}

static constexpr absl::string_view kRootParameterReplicatedSubmeshHlo = R"(
region_1.11 {
  Arg_0.12 = s32[] parameter(0)
  Arg_1.13 = s32[] parameter(1)
  ROOT add.14 = s32[] add(Arg_0.12, Arg_1.13)
}

region_2.15 {
  Arg_0.16 = s32[] parameter(0)
  Arg_1.17 = s32[] parameter(1)
  ROOT add.18 = s32[] add(Arg_0.16, Arg_1.17)
}

region_0.19 {
  arg_tuple.20 = (s32[], s32[], s32[4]{0}, s32[4]{0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  get-tuple-element.21 = s32[] get-tuple-element(arg_tuple.20), index=0
  constant.28 = s32[] constant(1)
  add.44 = s32[] add(get-tuple-element.21, constant.28)
  get-tuple-element.22 = s32[] get-tuple-element(arg_tuple.20), index=1
  constant.29 = s32[] constant(2)
  add.37 = s32[] add(get-tuple-element.22, constant.29)
  get-tuple-element.23 = s32[4]{0} get-tuple-element(arg_tuple.20), index=2
  get-tuple-element.25 = s32[4,4]{1,0} get-tuple-element(arg_tuple.20), index=4
  constant.31 = s32[] constant(0)
  compare.32 = pred[] compare(get-tuple-element.22, constant.31), direction=LT
  constant.30 = s32[] constant(4)
  add.33 = s32[] add(get-tuple-element.22, constant.30)
  select.34 = s32[] select(compare.32, add.33, get-tuple-element.22)
  dynamic-slice.35 = s32[2,4]{1,0} dynamic-slice(get-tuple-element.25, select.34, constant.31), dynamic_slice_sizes={2,4}
  custom-call.36 = s32[2,4]{1,0} custom-call(dynamic-slice.35), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  get-tuple-element.26 = s32[4,4]{1,0} get-tuple-element(arg_tuple.20), index=5
  dot.38 = s32[2,4]{1,0} dot(custom-call.36, get-tuple-element.26), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer0"}
  reduce.39 = s32[4]{0} reduce(dot.38, constant.31), dimensions={0}, to_apply=region_1.11
  add.42 = s32[4]{0} add(get-tuple-element.23, reduce.39), sharding={devices=[2]0,1}, metadata={op_name="layer0"}
  get-tuple-element.24 = s32[4]{0} get-tuple-element(arg_tuple.20), index=3
  get-tuple-element.27 = s32[4,4]{1,0} get-tuple-element(arg_tuple.20), index=6
  dot.40 = s32[2,4]{1,0} dot(custom-call.36, get-tuple-element.27), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer1"}
  reduce.41 = s32[4]{0} reduce(dot.40, constant.31), dimensions={0}, to_apply=region_2.15
  add.43 = s32[4]{0} add(get-tuple-element.24, reduce.41), sharding={replicated}, metadata={op_name="layer1"}
  ROOT tuple.45 = (s32[], s32[], s32[4]{0}, s32[4]{0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(add.44, add.37, add.42, add.43, get-tuple-element.25, /*index=5*/get-tuple-element.26, get-tuple-element.27)
} // region_0.19

region_3.46 {
  arg_tuple.47 = (s32[], s32[], s32[4]{0}, s32[4]{0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  get-tuple-element.49 = s32[] get-tuple-element(arg_tuple.47), index=1
  get-tuple-element.50 = s32[4]{0} get-tuple-element(arg_tuple.47), index=2
  get-tuple-element.51 = s32[4]{0} get-tuple-element(arg_tuple.47), index=3
  get-tuple-element.52 = s32[4,4]{1,0} get-tuple-element(arg_tuple.47), index=4
  get-tuple-element.53 = s32[4,4]{1,0} get-tuple-element(arg_tuple.47), index=5
  get-tuple-element.54 = s32[4,4]{1,0} get-tuple-element(arg_tuple.47), index=6
  get-tuple-element.48 = s32[] get-tuple-element(arg_tuple.47), index=0
  constant.55 = s32[] constant(2)
  ROOT compare.56 = pred[] compare(get-tuple-element.48, constant.55), direction=LT
} // region_3.46

ENTRY main.66 {
  constant.4 = s32[] constant(0)
  constant.5 = s32[] constant(0)
  broadcast.6 = s32[4]{0} broadcast(constant.5), dimensions={}
  custom-call.7 = s32[4]{0} custom-call(broadcast.6), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  custom-call.8 = s32[4]{0} custom-call(broadcast.6), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_0.1 = s32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  custom-call.0 = s32[4,4]{1,0} custom-call(Arg_0.1), custom_call_target="AutoSharding", backend_config={"axes": [["batch"], ["model"]]}
  custom-call.9 = s32[4,4]{1,0} custom-call(custom-call.0), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_1.2 = s32[4,4]{1,0} parameter(1)
  Arg_2.3 = s32[4,4]{1,0} parameter(2)
  tuple.10 = (s32[], s32[], s32[4]{0}, s32[4]{0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(constant.4, constant.4, custom-call.7, custom-call.8, custom-call.9, /*index=5*/Arg_1.2, Arg_2.3)
  while.57 = (s32[], s32[], s32[4]{0}, s32[4]{0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}) while(tuple.10), condition=region_3.46, body=region_0.19
  get-tuple-element.58 = s32[] get-tuple-element(while.57), index=0
  get-tuple-element.59 = s32[] get-tuple-element(while.57), index=1
  get-tuple-element.62 = s32[4,4]{1,0} get-tuple-element(while.57), index=4
  get-tuple-element.63 = s32[4,4]{1,0} get-tuple-element(while.57), index=5
  get-tuple-element.64 = s32[4,4]{1,0} get-tuple-element(while.57), index=6
  get-tuple-element.60 = s32[4]{0} get-tuple-element(while.57), index=2
  get-tuple-element.61 = s32[4]{0} get-tuple-element(while.57), index=3
  ROOT tuple.65 = (s32[4]{0}, s32[4]{0}) tuple(get-tuple-element.60, get-tuple-element.61)
} // main.66
)";

TEST_F(MpmdPartitionTest, RootParameterReplicatedSubmesh) {
  RegisterNamedTestTask("layer0", {0, 2}, {2}, {"x"}, {{"batch", "x"}});
  RegisterNamedTestTask("layer1", {2, 4}, {2}, {"x"}, {{"batch", "x"}});
  // just validate that it builds
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          RunMpmdOnHloString(kRootParameterReplicatedSubmeshHlo,
                                             /*num_devices=*/4));
  auto [tasks, intermediates] = std::move(result);
}

static constexpr absl::string_view kShardedDynamicSliceHlo = R"(
HloModule jit_wrapped, entry_computation_layout={(s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0})->(s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0})}, allow_spmd_sharding_propagation_to_output={true,true,true,true}, allow_spmd_sharding_propagation_to_parameters={true,true,true,true}

region_1.14 {
  Arg_0.15 = s32[] parameter(0)
  Arg_1.16 = s32[] parameter(1)
  ROOT add.17 = s32[] add(Arg_0.15, Arg_1.16)
}

region_0.18 {
  arg_tuple.19 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  get-tuple-element.20 = s32[] get-tuple-element(arg_tuple.19), index=0
  constant.30 = s32[] constant(1)
  add.54 = s32[] add(get-tuple-element.20, constant.30)
  get-tuple-element.21 = s32[] get-tuple-element(arg_tuple.19), index=1
  constant.31 = s32[] constant(2)
  add.39 = s32[] add(get-tuple-element.21, constant.31)
  get-tuple-element.22 = s32[] get-tuple-element(arg_tuple.19), index=2
  get-tuple-element.26 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=6
  constant.33 = s32[] constant(0)
  compare.34 = pred[] compare(get-tuple-element.21, constant.33), direction=LT
  constant.32 = s32[] constant(4)
  add.35 = s32[] add(get-tuple-element.21, constant.32)
  select.36 = s32[] select(compare.34, add.35, get-tuple-element.21)
  dynamic-slice.37 = s32[2,4]{1,0} dynamic-slice(get-tuple-element.26, select.36, constant.33), dynamic_slice_sizes={2,4}
  custom-call.38 = s32[2,4]{1,0} custom-call(dynamic-slice.37), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  get-tuple-element.27 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=7
  dot.40 = s32[2,4]{1,0} dot(custom-call.38, get-tuple-element.27), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer0"}
  get-tuple-element.28 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=8
  dot.41 = s32[2,4]{1,0} dot(dot.40, get-tuple-element.28), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer1"}
  get-tuple-element.29 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=9
  dot.42 = s32[2,4]{1,0} dot(dot.41, get-tuple-element.29), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="layer2"}
  reduce.43 = s32[] reduce(dot.42, constant.33), dimensions={0,1}, to_apply=region_1.14
  add.50 = s32[] add(get-tuple-element.22, reduce.43)
  get-tuple-element.23 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=3
  broadcast.44 = s32[4,4]{1,0} broadcast(reduce.43), dimensions={}
  multiply.45 = s32[4,4]{1,0} multiply(get-tuple-element.27, broadcast.44)
  add.51 = s32[4,4]{1,0} add(get-tuple-element.23, multiply.45)
  get-tuple-element.24 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=4
  broadcast.46 = s32[4,4]{1,0} broadcast(reduce.43), dimensions={}
  multiply.47 = s32[4,4]{1,0} multiply(get-tuple-element.28, broadcast.46)
  add.52 = s32[4,4]{1,0} add(get-tuple-element.24, multiply.47)
  get-tuple-element.25 = s32[4,4]{1,0} get-tuple-element(arg_tuple.19), index=5
  broadcast.48 = s32[4,4]{1,0} broadcast(reduce.43), dimensions={}
  multiply.49 = s32[4,4]{1,0} multiply(get-tuple-element.29, broadcast.48)
  add.53 = s32[4,4]{1,0} add(get-tuple-element.25, multiply.49)
  ROOT tuple.55 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(add.54, add.39, add.50, add.51, add.52, /*index=5*/add.53, get-tuple-element.26, get-tuple-element.27, get-tuple-element.28, get-tuple-element.29)
} // region_0.18

region_2.56 {
  arg_tuple.57 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  get-tuple-element.59 = s32[] get-tuple-element(arg_tuple.57), index=1
  get-tuple-element.60 = s32[] get-tuple-element(arg_tuple.57), index=2
  get-tuple-element.61 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=3
  get-tuple-element.62 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=4
  get-tuple-element.63 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=5
  get-tuple-element.64 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=6
  get-tuple-element.65 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=7
  get-tuple-element.66 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=8
  get-tuple-element.67 = s32[4,4]{1,0} get-tuple-element(arg_tuple.57), index=9
  get-tuple-element.58 = s32[] get-tuple-element(arg_tuple.57), index=0
  constant.68 = s32[] constant(2)
  ROOT compare.69 = pred[] compare(get-tuple-element.58, constant.68), direction=LT
} // region_2.56

ENTRY main.82 {
  constant.5 = s32[] constant(0)
  custom-call.8 = s32[] custom-call(constant.5), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  constant.6 = s32[] constant(0)
  broadcast.7 = s32[4,4]{1,0} broadcast(constant.6), dimensions={}
  custom-call.9 = s32[4,4]{1,0} custom-call(broadcast.7), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  custom-call.10 = s32[4,4]{1,0} custom-call(broadcast.7), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  custom-call.11 = s32[4,4]{1,0} custom-call(broadcast.7), custom_call_target="MicrobatchInit", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_0.1 = s32[4,4]{1,0} parameter(0), sharding={replicated}
  custom-call.3 = s32[4,4]{1,0} custom-call(Arg_0.1), custom_call_target="AutoSharding", backend_config={"axes": [["x"], ["y"]]}, metadata={op_name="task"}
  custom-call.12 = s32[4,4]{1,0} custom-call(custom-call.3), custom_call_target="Microbatch", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2}
  Arg_1.2 = s32[4,4]{1,0} parameter(1), sharding={replicated}
  Arg_2.3 = s32[4,4]{1,0} parameter(2), sharding={replicated}
  Arg_3.4 = s32[4,4]{1,0} parameter(3), sharding={replicated}
  tuple.13 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(constant.5, constant.5, custom-call.8, custom-call.9, custom-call.10, /*index=5*/custom-call.11, custom-call.12, Arg_1.2, Arg_2.3, Arg_3.4)
  while.70 = (s32[], s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, /*index=5*/s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) while(tuple.13), condition=region_2.56, body=region_0.18
  get-tuple-element.71 = s32[] get-tuple-element(while.70), index=0
  get-tuple-element.72 = s32[] get-tuple-element(while.70), index=1
  get-tuple-element.77 = s32[4,4]{1,0} get-tuple-element(while.70), index=6
  get-tuple-element.78 = s32[4,4]{1,0} get-tuple-element(while.70), index=7
  get-tuple-element.79 = s32[4,4]{1,0} get-tuple-element(while.70), index=8
  get-tuple-element.80 = s32[4,4]{1,0} get-tuple-element(while.70), index=9
  get-tuple-element.73 = s32[] get-tuple-element(while.70), index=2
  get-tuple-element.74 = s32[4,4]{1,0} get-tuple-element(while.70), index=3
  get-tuple-element.75 = s32[4,4]{1,0} get-tuple-element(while.70), index=4
  get-tuple-element.76 = s32[4,4]{1,0} get-tuple-element(while.70), index=5
  ROOT tuple.81 = (s32[], s32[4,4]{1,0}, s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(get-tuple-element.73, get-tuple-element.74, get-tuple-element.75, get-tuple-element.76)
}
)";

TEST_F(MpmdPartitionTest, ShardedDynamicSlice) {
  RegisterNamedTestTask("layer0", {0, 2}, {2}, {"x"}, {{"x", "x"}});
  RegisterNamedTestTask("layer1", {2, 4}, {2}, {"x"}, {{"x", "x"}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kShardedDynamicSliceHlo,
                         /*num_devices=*/4,
                         {.use_module_config_auto_param_sharding = true,
                          .use_auto_input_sharding = true}));
  auto [tasks, intermediates] = std::move(result);

  Summarize(tasks);

  EXPECT_THAT(tasks,
              Each(AllOf(
                  // all non-trivial intermediates should be sharded
                  m::TaskInputs(Not(Contains(
                      AllOf(m::Store({.type = Store::Type::TEMP}),
                            NonTrivialStoreShape(), TrivialStoreSharding())))),
                  m::TaskOutputs(Not(Contains(AllOf(
                      m::Store({.type = Store::Type::TEMP}),
                      NonTrivialStoreShape(), TrivialStoreSharding())))))));
}

static constexpr absl::string_view kHloAutoShardingDotHlo = R"(
ENTRY main.45 {
  Arg_0.1 = bf16[32,2048,12288]{2,1,0} parameter(0)
  Arg_1.2 = bf16[32,2048,3072]{2,1,0} parameter(1)
  Arg_2.3 = bf16[12288,3072]{1,0} parameter(2)
  custom-call.3 = bf16[12288,3072]{1,0} custom-call(Arg_2.3), custom_call_target="AutoSharding", backend_config={"axes": [["x"], ["y"]]}, metadata={op_name="task1"}
  add.5895 = bf16[32,2048,12288]{2,1,0} add(Arg_0.1, Arg_0.1), sharding={devices=[4,1,1]0,1,2,3}
  add.5844 = bf16[32,2048,3072]{2,1,0} add(Arg_1.2, Arg_1.2), sharding={devices=[4,1,1]0,1,2,3}
  dot.5900 = bf16[12288,3072]{1,0} dot(add.5895, add.5844), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,1}
  add.1 = bf16[12288,3072]{1,0} add(custom-call.3, dot.5900), metadata={op_name="task2"}
  ROOT tuple = (bf16[12288,3072]{1,0}, bf16[12288,3072]{1,0}) tuple(add.1, add.1)
}
)";

static constexpr absl::string_view kAutoShardingDotShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 2
iota_reshape_dims: 4
iota_transpose_perm: 0
)";

TEST_F(MpmdPartitionTest, HloAutoshardingDotHlo) {
  GTEST_SKIP() << "autosharding from dot operands still needs to be defined";

  RegisterNamedTestTask("task1", {0, 4}, {2, 2}, {"x", "y"},
                        {{"x", "x"}, {"y", "y"}});
  RegisterNamedTestTask("task2", {4, 8}, {2, 2}, {"x", "y"},
                        {{"x", "x"}, {"y", "y"}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kHloAutoShardingDotHlo,
                         /*num_devices=*/8, {.use_auto_input_sharding = true}));
  auto [tasks, intermediates] = std::move(result);

  TF_ASSERT_OK_AND_ASSIGN(HloSharding dot_sharding,
                          GetSharding(kAutoShardingDotShardingPbtxt));
  HloSharding arg_sharding = dot_sharding;
}

static constexpr absl::string_view kRematOptBarrierSubsetTupleHlo = R"(
ENTRY main.35 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  add.1 = f32[8]{0} add(Arg_0.1, Arg_0.1), sharding={devices=[2]0,1}
  add.2 = f32[8]{0} add(Arg_0.1, add.1), sharding={devices=[2]0,1}
  add.3 = f32[8]{0} add(add.2, add.1), sharding={devices=[2]2,3}
  add.4 = f32[8]{0} add(add.3, add.1), sharding={devices=[2]2,3}
  tuple.3 = (f32[8]{0}, f32[8]{0}, f32[8]{0}, f32[8]{0}) tuple(add.1,add.2,add.3,add.4)
  opt-barrier.4 = (f32[8]{0}, f32[8]{0}, f32[8]{0}, f32[8]{0}) opt-barrier(tuple.3)
  element.5 = f32[8]{0} get-tuple-element(opt-barrier.4), index=0
  element.6 = f32[8]{0} get-tuple-element(opt-barrier.4), index=1
  element.7 = f32[8]{0} get-tuple-element(opt-barrier.4), index=2
  element.8 = f32[8]{0} get-tuple-element(opt-barrier.4), index=3
  add.7 = f32[8]{0} add(element.5, element.6), sharding={devices=[2]0,1}
  add.8 = f32[8]{0} add(element.7, element.8), sharding={devices=[2]2,3}
  ROOT tuple = (f32[8]{0}, f32[8]{0}) tuple(add.7, add.8)
} // main.35
)";

TEST_F(MpmdPartitionTest, RematOptBarrierSubsetTupleHlo) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kRematOptBarrierSubsetTupleHlo,
                         /*num_devices=*/4, {.use_auto_input_sharding = true}));
  auto [tasks, intermediates] = std::move(result);

  // each user task of the barrier should have a subset optimization barrier
  // with 2 inputs
  EXPECT_THAT(tasks,
              Contains(m::TaskInstructions(Contains(
                  op::OptimizationBarrier(op::Tuple(op::Add(), op::Add()))))));
}

TEST_F(MpmdPartitionTest, RematBackpropMultipleLayers) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, RunMpmdOnHloTextPath("remat_backprop_multiple_layers.txt",
                                        /*num_devices=*/1,
                                        {.use_auto_input_sharding = true}));
  auto [tasks, intermediates] = std::move(result);
}

TEST_F(MpmdPartitionTest, ReshapeMicrobatchInit) {
  // This should crash with an error about over-sharding
  // a given dimension
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, RunMpmdOnHloTextPath("8gpu_spmd_failure.txt",
                                        /*num_devices=*/8,
                                        {.use_auto_input_sharding = false}));
}

TEST_F(MpmdPartitionTest, LoopInputInstruction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, RunMpmdOnHloTextPath("microbatch_pre_post.txt",
                                        /*num_devices=*/1,
                                        {.use_auto_input_sharding = true}));
  // just validate that it gets built
  auto [tasks, intermediates] = std::move(result);
}

TEST_F(MpmdPartitionTest, DecomposeMultipleLayers) {
  EnableMultiMeshRecomputation(true);
  auto device_factory = [](const std::string& name, bool backprop) {
    int layer_num;
    bool parsed = absl::SimpleAtoi(name.substr(9), &layer_num);
    if (!parsed) {
      throw std::runtime_error(
          absl::StrCat("failed to parse layer number from", name));
    }
    int64_t offset = (layer_num / 4) * 4;
    std::string color = [&] {
      if (backprop) {
        return absl::StrCat("bwd.", name);
      }
      return name;
    }();
    std::vector<int64_t> devices(4);
    std::iota(devices.begin(), devices.end(), offset);
    return std::make_pair(std::move(devices), std::move(color));
  };
  RegisterMatcherTestTaskWithFactory("(x_layers_\\d+)", device_factory, {2, 2},
                                     {"x", "y"},
                                     {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(emb).*", {0, 8}, {2, 4}, {"x", "y"},
                          {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(final_ln).*", {0, 8}, {2, 4}, {"x", "y"},
                          {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(compute_loss).*", {0, 8}, {2, 4}, {"x", "y"},
                          {{"replica", "x"}, {"mdl", "y"}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloTextPath(
          "opt_8layers.txt",
          /*num_devices=*/8,
          {
              .use_auto_input_sharding = true,
              .replicated_parameter_num_elements_cutoff = 1024 * 1024,
              .recompute_from_arguments_if_cost_less_than = 1024 * 1024,
          }));
  auto [tasks, intermediates] = std::move(result);

  // if the task has a dynamic slice, it should have a slice input
  EXPECT_THAT(tasks,
              Each(Not(AllOf(m::TaskInstructions(Contains(op::DynamicSlice())),
                             m::LoopTask(false)))));
}

TEST_F(MpmdPartitionTest, AutoshardingRootReturn) {
  // just validate that it builds
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          RunMpmdOnHloTextPath("autosharding_root.txt",
                                               /*num_devices=*/4,
                                               {.use_auto_input_sharding = true,
                                                .only_fuse_loop_tasks = true}));
  auto [tasks, intermediates] = std::move(result);
}

TEST_F(MpmdPartitionTest, DecomposeMicrobatchSpmd) {
  EnableMultiMeshRecomputation(true);
  RegisterMatcherTestTask("(x_layers_\\d+).*", {0, 2}, {2, 1}, {"x", "y"},
                          {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(position_emb).*", {0, 2}, {2, 1}, {"x", "y"},
                          {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(emb_lookup).*", {0, 2}, {2, 1}, {"x", "y"},
                          {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(final_ln).*", {0, 2}, {2, 1}, {"x", "y"},
                          {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(compute_loss).*", {0, 2}, {2, 1}, {"x", "y"},
                          {{"replica", "x"}, {"mdl", "y"}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result, RunMpmdOnHloTextPath("spmd_microbatch_4layers.txt",
                                        /*num_devices=*/2,
                                        {.use_auto_input_sharding = true}));
  auto [tasks, intermediates] = std::move(result);
}

TEST_F(MpmdPartitionTest, ReplicateArgumentThroughLoop) {
  EnableMultiMeshRecomputation(true);
  RegisterMatcherTestTask(
      "(x_layers_\\d+).*", {0, 2}, {1, 1, 2}, {"x", "y", "z"},
      {{"replica", "x"}, {"data", "y"}, {"mdl", "z"}, {"seq", "z"}});
  RegisterMatcherTestTask(
      "(emb).*", {0, 2}, {1, 1, 2}, {"x", "y", "z"},
      {{"replica", "x"}, {"data", "y"}, {"mdl", "z"}, {"seq", "z"}});
  RegisterMatcherTestTask(
      "(final_ln).*", {0, 2}, {1, 1, 2}, {"x", "y", "z"},
      {{"replica", "x"}, {"data", "y"}, {"mdl", "z"}, {"seq", "z"}});
  RegisterMatcherTestTask(
      "(compute_loss).*", {0, 2}, {1, 1, 2}, {"x", "y", "z"},
      {{"replica", "x"}, {"data", "y"}, {"mdl", "z"}, {"seq", "z"}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloTextPath(
          "replicated_sharding_through_loop.txt",
          /*num_devices=*/2,
          {.use_auto_input_sharding = true,
           .recompute_from_arguments_if_cost_less_than = 1024 * 1024,
           .replicated_parameter_num_elements_cutoff = 8 * 2048}));
  auto [tasks, intermediates] = std::move(result);
}

TEST_F(MpmdPartitionTest, RngColoringFail) {
  EnableMultiMeshRecomputation(true);
  RegisterMatcherTestTask(
      "(layers_\\d+).*", {0, 2}, {1, 1, 2}, {"x", "y", "z"},
      {{"replica", "x"}, {"data", "y"}, {"mdl", "z"}, {"seq", "z"}});
  RegisterMatcherTestTask(
      "(emb).*", {0, 2}, {1, 1, 2}, {"x", "y", "z"},
      {{"replica", "x"}, {"data", "y"}, {"mdl", "z"}, {"seq", "z"}});
  RegisterMatcherTestTask(
      "(final_ln).*", {0, 2}, {1, 1, 2}, {"x", "y", "z"},
      {{"replica", "x"}, {"data", "y"}, {"mdl", "z"}, {"seq", "z"}});
  RegisterMatcherTestTask(
      "(compute_loss).*", {0, 2}, {1, 1, 2}, {"x", "y", "z"},
      {{"replica", "x"}, {"data", "y"}, {"mdl", "z"}, {"seq", "z"}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloTextPath(
          "rng_coloring_fail.txt",
          /*num_devices=*/2,
          {.use_auto_input_sharding = true,
           .recompute_from_arguments_if_cost_less_than = 1024 * 1024,
           .replicated_parameter_num_elements_cutoff = 8 * 2048}));

  // just validate that it completes
  auto [tasks, intermediates] = std::move(result);
}

TEST_F(MpmdPartitionTest, InitModelFromSeedFailure) {
  EnableMultiMeshRecomputation(true);
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloTextPath(
          "init_model_from_seed_failure.txt",
          /*num_devices=*/8,
          {.use_auto_input_sharding = true,
           .recompute_from_arguments_if_cost_less_than = 1024 * 1024,
           .replicated_parameter_num_elements_cutoff = 8 * 2048}));

  // just validate that it completes
  auto [tasks, intermediates] = std::move(result);
}

}  // namespace
}  // namespace xla
