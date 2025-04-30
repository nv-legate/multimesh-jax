/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_logical_to_gspmd_sharding.h"

#include <utility>

#include "gmock/gmock.h"
#include "tsl/platform/regexp.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/logical_sharding_context.h"
#include "xla/pjrt/legate/mpmd_coloring.h"
#include "xla/pjrt/legate/mpmd_computation_grouper.h"
#include "xla/pjrt/legate/mpmd_computation_inliner.h"
#include "xla/pjrt/legate/mpmd_logical_sharding_propagation.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdLogicalShardingTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

using ::testing::Each;
using ::testing::ElementsAre;

static constexpr absl::string_view kSimpleAutoshardedHlo = R"(
ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0)
  custom-call = f32[8]{0} custom-call(Arg_0.1), custom_call_target="AutoSharding", backend_config={"axes": [["x"]]}
  multiply.4 = f32[8]{0} multiply(custom-call, custom-call), metadata={op_name="task_f"}
  multiply.7 = f32[8]{0} multiply(multiply.4, multiply.4), metadata={op_name="task_f"}
  cosine.8 = f32[8]{0} cosine(multiply.7)
  custom-call.1 = f32[8]{0} custom-call(cosine.8), custom_call_target="AutoSharding", backend_config={"axes": [["x"]]}
  add.9 = f32[8]{0} add(cosine.8, Arg_0.1), metadata={op_name="task_g"}
  constant.3 = f32[] constant(0)
  broadcast.3 = f32[8]{0} broadcast(constant.3), dimensions={}
  ROOT add.14 = f32[] add(add.9, broadcast.3)
} // main.15
)";

static constexpr absl::string_view kTask0Sharding = R"(
type: OTHER
tile_assignment_dimensions: 2
iota_reshape_dims: 2
iota_transpose_perm: 0
iota_offset: 0
)";

static constexpr absl::string_view kTask1Sharding = R"(
type: OTHER
tile_assignment_dimensions: 2
iota_reshape_dims: 2
iota_transpose_perm: 0
iota_offset: 2
)";

TEST_F(MpmdLogicalShardingTest, SimpleAutosharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kSimpleAutoshardedHlo, /*num_devices=*/4));

  RegisterMatcherTestTask("(task_f)", {0, 2}, {2}, {"x"}, {{"x", "x"}});
  RegisterMatcherTestTask("(task_g)", {2, 4}, {2}, {"x"}, {{"x", "x"}});

  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  MpmdLogicalShardingPropagation propagation;
  TF_ASSERT_OK_AND_ASSIGN(changed, propagation.Run(module.get()));

  // autosharding expects the module to be in grouped form
  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(changed, grouper.Run(module.get()));

  MpmdLogicalToGSPMDSharding autosharding{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(changed, autosharding.Run(module.get()));

  std::vector<HloComputation*> calls;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      calls.push_back(instruction->called_computations()[0]);
    }
  }

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sh0, GetSharding(kTask0Sharding));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding sh1, GetSharding(kTask1Sharding));

  // there should be a single parameter with the desired sharding
  EXPECT_THAT(calls[0]->instructions(),
              Contains(AllOf(op::Parameter(), op::Sharding(sh0))).Times(1));

  // there should be two parameters with the desired sharding
  EXPECT_THAT(calls[1]->instructions(),
              Contains(AllOf(op::Parameter(), op::Sharding(sh1))).Times(2));

  MpmdComputationInliner inliner{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(changed, inliner.Run(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(changed, grouper.Run(module.get()));
}

static constexpr absl::string_view kTransposeOperandAutoShardHlo = R"(
ENTRY main.7 {
  Arg_0.1 = s32[4,2,3]{2,1,0} parameter(0), sharding={replicated}
  add.1 = s32[4,2,3]{2,1,0} add(Arg_0.1, Arg_0.1)
  custom-call.3 = s32[4,2,3]{2,1,0} custom-call(add.1), custom_call_target="AutoSharding", backend_config={"axes": [["x"], ["y"], []]}, metadata={op_name="task"}
  transpose.2 = s32[2,3,4]{2,1,0} transpose(custom-call.3), dimensions={1,2,0}
  ROOT multiply.4 = s32[2,3,4]{2,1,0} multiply(transpose.2, transpose.2), metadata={op_name="task"}
} // main.7
)";
static constexpr absl::string_view kTransposeOperandAutoShardShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 1
tile_assignment_dimensions: 4
iota_reshape_dims: 4
iota_reshape_dims: 2
iota_transpose_perm: 1
iota_transpose_perm: 0
)";

TEST_F(MpmdLogicalShardingTest, TransposeOperandAutoShardParam) {
  GTEST_SKIP() << "autosharding propagation across kTranspose not yet defined";

  RegisterMatcherTestTask("(task)", {0, 8}, {4, 2, 1}, {"x", "y", "z"},
                          {{"x", "x"}, {"y", "y"}, {"z", "z"}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kTransposeOperandAutoShardHlo, /*num_devices=*/8,
                         {.use_auto_input_sharding = true}));

  auto [sharded_tasks, sharded_intermediates] = std::move(result);

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding,
                          GetSharding(kTransposeOperandAutoShardShardingPbtxt));

  EXPECT_THAT(
      sharded_tasks,
      ElementsAre(m::TaskRoots(Contains(op::Sharding(sharding)).Times(1))));
}

static constexpr absl::string_view kTransposeAutoShardHlo = R"(
ENTRY main.7 {
  Arg_0.1 = s32[4,2,3]{2,1,0} parameter(0)
  add.1 = s32[4,2,3]{2,1,0} add(Arg_0.1, Arg_0.1), metadata={op_name="task"}
  transpose.2 = s32[2,3,4]{2,1,0} transpose(add.1), dimensions={1,2,0}, metadata={op_name="task"}
  custom-call.3 = s32[2,3,4]{2,1,0} custom-call(transpose.2), custom_call_target="AutoSharding", backend_config={"axes": [["x"], [], ["z"]]}, metadata={op_name="task"}
  ROOT multiply.4 = s32[2,3,4]{2,1,0} multiply(custom-call.3, custom-call.3), metadata={op_name="task"}
} // main.7
)";
static constexpr absl::string_view kTransposeAutoShardShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 4
tile_assignment_dimensions: 2
tile_assignment_dimensions: 1
iota_reshape_dims: 2
iota_reshape_dims: 4
iota_transpose_perm: 1
iota_transpose_perm: 0
)";

TEST_F(MpmdLogicalShardingTest, TransposeAutoShardParam) {
  GTEST_SKIP() << "autosharding propagation across kTranspose not yet defined";

  RegisterMatcherTestTask("(task)", {0, 8}, {2, 1, 4}, {"x", "y", "z"},
                          {{"x", "x"}, {"y", "y"}, {"z", "z"}});

  TF_ASSERT_OK_AND_ASSIGN(
      auto result, RunMpmdOnHloString(kTransposeAutoShardHlo, /*num_devices=*/8,
                                      {.use_auto_input_sharding = true}));

  auto [sharded_tasks, sharded_intermediates] = std::move(result);

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding,
                          GetSharding(kTransposeAutoShardShardingPbtxt));

  EXPECT_THAT(sharded_tasks,
              ElementsAre(AllOf(
                  m::TaskParameters(Contains(op::Sharding(sharding)).Times(1)),
                  m::TaskInstructions(Not(Contains(op::CustomCall()))))));
}

static constexpr absl::string_view kSimpleAutoShardParamHlo = R"(
ENTRY main.7 {
  Arg_0.1 = s32[4,2]{1,0} parameter(0)
  custom-call.3 = s32[4,2]{1,0} custom-call(Arg_0.1), custom_call_target="AutoSharding", backend_config={"axes": [["x"], ["y"]]}, metadata={op_name="task"}
  ROOT multiply.4 = s32[4,2]{1,0} multiply(custom-call.3, custom-call.3), metadata={op_name="task"}
} // main.7
)";
static constexpr absl::string_view kSimpleAutoShardParamInputShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 1
iota_reshape_dims: 2
iota_transpose_perm: 0
)";

TEST_F(MpmdLogicalShardingTest, SimpleAutoShardParam) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kSimpleAutoShardParamHlo, /*num_devices=*/2,
                         {.use_auto_input_sharding = false}));
  auto [unsharded_tasks, unsharded_intermediates] = std::move(result);

  // even though the no autosharding context was registered,
  // the custom call should have been removed
  EXPECT_THAT(
      unsharded_tasks,
      ElementsAre(m::TaskInstructions(Not(Contains(op::CustomCall())))));

  RegisterMatcherTestTask("(task)", {0, 2}, {2, 1}, {"x", "y"},
                          {{"x", "x"}, {"y", "y"}});
  TF_ASSERT_OK_AND_ASSIGN(
      result, RunMpmdOnHloString(kSimpleAutoShardParamHlo, /*num_devices=*/2,
                                 {.use_auto_input_sharding = true}));
  auto [sharded_tasks, sharded_intermediates] = std::move(result);

  // autosharding custom call should be gone
  EXPECT_THAT(
      sharded_tasks,
      ElementsAre(m::TaskInstructions(Not(Contains(op::CustomCall())))));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding,
                          GetSharding(kSimpleAutoShardParamInputShardingPbtxt));

  EXPECT_THAT(sharded_tasks, ElementsAre(m::TaskParameters(
                                 Contains(op::Sharding(sharding)).Times(1))));
}

static constexpr absl::string_view kGradientAutoShardParamHlo = R"(
region_0.9 {
  Arg_0.10 = f32[] parameter(0)
  Arg_1.11 = f32[] parameter(1)
  ROOT add.12 = f32[] add(Arg_0.10, Arg_1.11), metadata={op_name="task_1"}
}

ENTRY main.19 {
  Arg_0.1 = f32[4,2]{1,0} parameter(0), sharding={replicated}
  custom-call.4 = f32[4,2]{1,0} custom-call(Arg_0.1), custom_call_target="AutoSharding", metadata={op_name="task_0"}, backend_config={"axes": [["x"], ["y"]]}
  Arg_1.2 = s32[] parameter(1), sharding={replicated}
  convert.5 = f32[] convert(Arg_1.2), metadata={op_name="task_0"}
  broadcast.6 = f32[4,2]{1,0} broadcast(convert.5), dimensions={}, metadata={op_name="task_0"}
  multiply.7 = f32[4,2]{1,0} multiply(custom-call.4, broadcast.6), metadata={op_name="task_0"}
  multiply.77 = f32[4,2]{1,0} multiply(multiply.7, broadcast.6), metadata={op_name="task_0"}
  multiply.8 = f32[4,2]{1,0} multiply(multiply.77, multiply.77), metadata={op_name="task_1"}
  constant.3 = f32[] constant(0)
  reduce.13 = f32[] reduce(multiply.8, constant.3), dimensions={0,1}, to_apply=region_0.9, metadata={op_name="task_1"}
  add.14 = f32[4,2]{1,0} add(multiply.77, multiply.77), metadata={op_name="task_1"}
  broadcast.15 = f32[4,2]{1,0} broadcast(convert.5), dimensions={}, metadata={op_name="task_0"}
  multiply.16 = f32[4,2]{1,0} multiply(add.14, broadcast.15), metadata={op_name="task_0"}
  custom-call.17 = f32[4,2]{1,0} custom-call(multiply.16), custom_call_target="AutoSharding", metadata={op_name="task_0"}, backend_config={"axes": [["x"], ["y"]]}
  ROOT tuple.18 = (f32[], f32[4,2]{1,0}) tuple(reduce.13, custom-call.17)
} // main.19
)";

TEST_F(MpmdLogicalShardingTest, GradientAutoShardParam) {
  static constexpr int kDevicesPerTask = 2;
  auto device_factory = [=](const std::string& task, bool backprop) {
    std::string task_number_str;
    bool match = RE2::FullMatch(task, "task_(\\d+)", &task_number_str);
    if (!match) {
      std::cerr << "Failed parsing task " << task
                << ", this should not have happened" << std::endl;
      abort();
    }

    int task_number;
    if (!absl::SimpleAtoi(task_number_str, &task_number)) {
      std::cerr << "Failed parsing numerical " << task_number_str
                << ", this should not have happened" << std::endl;
      abort();
    }

    std::string color = [&] {
      if (backprop) {
        return absl::StrCat("bwd.", task);
      }
      return task;
    }();
    int offset = task_number * kDevicesPerTask;
    auto devices = std::make_pair(offset, offset + kDevicesPerTask);
    return std::make_pair(devices, std::move(color));
  };

  RegisterMatcherTestTaskWithFactory("(task_\\d+)", std::move(device_factory),
                                     {2, 1}, {"x", "y"},
                                     {{"x", "x"}, {"y", "y"}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kGradientAutoShardParamHlo, /*num_devices=*/4,
                         {.use_auto_input_sharding = true}));
  auto [sharded_tasks, sharded_intermediates] = std::move(result);

  // make sure that all the temporaries get the correct sharding even
  // when the sharding propagation is not allowed to the output
  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding,
                          GetSharding(kSimpleAutoShardParamInputShardingPbtxt));

  EXPECT_THAT(
      sharded_tasks,
      Each(AllOf(m::TaskRoots(Each(m::ShardingOrReplicated(sharding))),
                 m::TaskParameters(Each(m::ShardingOrReplicated(sharding))))));
}

static constexpr absl::string_view kGradientMultiTaskAutoShardHlo = R"(
HloModule jit_c, allow_spmd_sharding_propagation_to_parameters={true,true,true}
ENTRY main.33 {
  Arg_0.1 = f32[4,2]{1,0} parameter(0), sharding={replicated}
  Arg_1.2 = f32[4,2]{1,0} parameter(1), sharding={replicated}
  custom-call.10 = f32[4,2]{1,0} custom-call(Arg_1.2), custom_call_target="AutoSharding", metadata={op_name="jit(step)/jit(main)/jvp(task_1)/AutoSharding/AutoSharding"}, backend_config={"axes": [["x"], ["y"]]}
  Arg_2.3 = s32[] parameter(2), sharding={replicated}
  convert.11 = f32[] convert(Arg_2.3), metadata={op_name="jit(step)/jit(main)/jvp(task_1)/convert_element_type[new_dtype=float32 weak_type=False]"}
  broadcast.12 = f32[4,2]{1,0} broadcast(convert.11), dimensions={}, metadata={op_name="jit(step)/jit(main)/jvp(task_1)/mul"}
  multiply.13 = f32[4,2]{1,0} multiply(custom-call.10, broadcast.12), metadata={op_name="jit(step)/jit(main)/jvp(task_1)/mul"}
  cosine.16 = f32[4,2]{1,0} cosine(multiply.13), metadata={op_name="jit(step)/jit(main)/jvp(task_1)/cos"}
  negate.23 = f32[4,2]{1,0} negate(cosine.16), metadata={op_name="jit(step)/jit(main)/transpose(jvp(task_1))/neg"}
  custom-call.6 = f32[4,2]{1,0} custom-call(Arg_0.1), custom_call_target="AutoSharding", metadata={op_name="jit(step)/jit(main)/jvp(task_0)/AutoSharding/AutoSharding"}, backend_config={"axes": [["x"], ["y"]]}
  convert.7 = f32[] convert(Arg_2.3), metadata={op_name="jit(step)/jit(main)/jvp(task_0)/convert_element_type[new_dtype=float32 weak_type=False]"}
  broadcast.8 = f32[4,2]{1,0} broadcast(convert.7), dimensions={}, metadata={op_name="jit(step)/jit(main)/jvp(task_0)/mul"}
  multiply.9 = f32[4,2]{1,0} multiply(custom-call.6, broadcast.8), metadata={op_name="jit(step)/jit(main)/jv/p(task_0)/mul"}
  sine.15 = f32[4,2]{1,0} sine(multiply.9), metadata={op_name="jit(step)/jit(main)/jvp(task_1)/sin"}
  multiply.24 = f32[4,2]{1,0} multiply(negate.23, sine.15), metadata={op_name="jit(step)/jit(main)/transpose(jvp(task_1))/mul"}
  broadcast.25 = f32[4,2]{1,0} broadcast(convert.7), dimensions={}, metadata={op_name="jit(step)/jit(main)/transpose(jvp(task_0))/mul"}
  multiply.26 = f32[4,2]{1,0} multiply(multiply.24, broadcast.25), metadata={op_name="jit(step)/jit(main)/transpose(jvp(task_0))/mul"}
  custom-call.27 = f32[4,2]{1,0} custom-call(multiply.26), custom_call_target="AutoSharding", metadata={op_name="jit(step)/jit(main)/transpose(jvp(task_0))/AutoSharding/AutoSharding"}, backend_config={"axes": [["x"], ["y"]]}
  constant.4 = f32[] constant(0.1)
  broadcast.5 = f32[4,2]{1,0} broadcast(constant.4), dimensions={}
  multiply.28 = f32[4,2]{1,0} multiply(custom-call.27, broadcast.5), metadata={op_name="jit(step)/jit(main)/mul"}
  subtract.29 = f32[4,2]{1,0} subtract(Arg_0.1, multiply.28), metadata={op_name="jit(step)/jit(main)/sub"}
  cosine.14 = f32[4,2]{1,0} cosine(multiply.9), metadata={op_name="jit(step)/jit(main)/jvp(task_1)/cos"}
  negate.18 = f32[4,2]{1,0} negate(cosine.14), metadata={op_name="jit(step)/jit(main)/transpose(jvp(task_1))/neg"}
  sine.17 = f32[4,2]{1,0} sine(multiply.13), metadata={op_name="jit(step)/jit(main)/jvp(task_1)/sin"}
  multiply.19 = f32[4,2]{1,0} multiply(negate.18, sine.17), metadata={op_name="jit(step)/jit(main)/transpose(jvp(task_1))/mul"}
  broadcast.20 = f32[4,2]{1,0} broadcast(convert.11), dimensions={}, metadata={op_name="jit(step)/jit(main)/transpose(jvp(task_1))/mul"}
  multiply.21 = f32[4,2]{1,0} multiply(multiply.19, broadcast.20), metadata={op_name="jit(step)/fjit(main)/transpose(jvp(task_1))/mul"}
  custom-call.22 = f32[4,2]{1,0} custom-call(multiply.21), custom_call_target="AutoSharding", metadata={op_name="jit(step)/jit(main)/transpose(jvp(task_1))/AutoSharding/AutoSharding"}, backend_config={"axes": [["x"], ["y"]]}
  multiply.30 = f32[4,2]{1,0} multiply(custom-call.22, broadcast.5), metadata={op_name="jit(step)/jit(main)/mul"}
  subtract.31 = f32[4,2]{1,0} subtract(Arg_1.2, multiply.30), metadata={op_name="jit(step)/jit(main)/sub"}
  ROOT tuple.32 = (f32[4,2]{1,0}, f32[4,2]{1,0}) tuple(subtract.29, subtract.31)
} // main.33
)";

TEST_F(MpmdLogicalShardingTest, GradientMultiTaskAutoShardHlo) {
  RegisterMatcherTestTask("(task_0)", {0, 2}, {2, 1}, {"x", "y"},
                          {{"x", "x"}, {"y", "y"}});
  RegisterMatcherTestTask("(task_1)", {2, 4}, {2, 1}, {"x", "y"},
                          {{"x", "x"}, {"y", "y"}});
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      RunMpmdOnHloString(kGradientMultiTaskAutoShardHlo, /*num_devices=*/4,
                         {.use_auto_input_sharding = true,
                          .use_module_config_auto_param_sharding = true}));
  auto [sharded_tasks, sharded_intermediates] = std::move(result);

  // make sure that all the inputs and outputs get the correct sharding
  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding,
                          GetSharding(kSimpleAutoShardParamInputShardingPbtxt));

  EXPECT_THAT(
      sharded_tasks,
      Each(AllOf(m::TaskRoots(Each(m::ShardingOrReplicated(sharding))),
                 m::TaskParameters(Each(m::ShardingOrReplicated(sharding))))));

  // the task_0 forward and backprop tasks should have devices 0,1
  // the task_1 fwd/bad tasks should have devices 2,3
  EXPECT_THAT(sharded_tasks, ElementsAre(m::TaskDevices(ElementsAre(0, 1)),
                                         m::TaskDevices(ElementsAre(2, 3)),
                                         m::TaskDevices(ElementsAre(2, 3)),
                                         m::TaskDevices(ElementsAre(0, 1))));
}

TEST_F(MpmdLogicalShardingTest, LargeParametersNotSharded) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("autoshard_large_parameters_not_sharded.txt",
                           /*num_devices=*/8));

  zuku::DeviceList devices04{{.start = 0, .num_devices = 4}};
  zuku::DeviceList devices48{{.start = 4, .num_devices = 4}};
  LogicalShardingContext context04{
      .devices = devices04,
      .dims = {1, 1, 4},
      .device_axes = {"x", "y", "z"},
      .logical_axes =
          {
              {"replica", "x"},
              {"data", "y"},
              {"mdl", "z"},
          },
  };
  auto context48 = context04;
  context48.devices = devices48;

  auto context04_ptr =
      std::make_shared<LogicalShardingContext>(std::move(context04));
  auto context48_ptr =
      std::make_shared<LogicalShardingContext>(std::move(context48));

  TF_ASSIGN_OR_RETURN(
      auto emb, partition_->AllocateColor("emb", devices04, context04_ptr));
  TF_ASSIGN_OR_RETURN(auto layer0, partition_->AllocateColor(
                                       "layers_0", devices04, context04_ptr));
  TF_ASSIGN_OR_RETURN(auto layer2, partition_->AllocateColor(
                                       "layers_2", devices04, context04_ptr));
  TF_ASSIGN_OR_RETURN(auto layer4, partition_->AllocateColor(
                                       "layers_4", devices04, context04_ptr));
  TF_ASSIGN_OR_RETURN(auto layer6, partition_->AllocateColor(
                                       "layers_6", devices04, context04_ptr));

  TF_ASSIGN_OR_RETURN(auto layer1, partition_->AllocateColor(
                                       "layers_1", devices48, context48_ptr));
  TF_ASSIGN_OR_RETURN(auto layer3, partition_->AllocateColor(
                                       "layers_3", devices48, context48_ptr));
  TF_ASSIGN_OR_RETURN(auto layer5, partition_->AllocateColor(
                                       "layers_5", devices48, context48_ptr));
  TF_ASSIGN_OR_RETURN(auto layer7, partition_->AllocateColor(
                                       "layers_7", devices48, context48_ptr));
  TF_ASSIGN_OR_RETURN(auto loss, partition_->AllocateColor(
                                     "compute_loss", devices48, context48_ptr));
  TF_ASSIGN_OR_RETURN(auto final_ln, partition_->AllocateColor(
                                         "final_ln", devices48, context48_ptr));

  MpmdLogicalToGSPMDSharding autoshard{partition_.get()};
  TF_ASSIGN_OR_RETURN(bool changed, autoshard.Run(module.get()));
}

}  // namespace
}  // namespace xla
