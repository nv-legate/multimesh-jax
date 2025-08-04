/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_insert_reshard.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::Field;
using ::testing::Not;

class MpmdInsertReshardTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

constexpr absl::string_view kBasicTasksHlo = R"(
task_f {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  param.1 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  add.0 = f32[4,4]{1,0} add(param.0, param.1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  add.1 = f32[4,4]{1,0} add(add.0, param.1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.0, add.1)
}

task_g {
  param.2 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.3 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.4 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.5  = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.4 = f32[4,4]{1,0} add(param.2, param.3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.5 = f32[4,4]{1,0} add(param.4, param.5), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.4 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.4, add.5)
}

task_init {
  zero = f32[] constant(0.0)
  broadcast.0 = f32[4,4]{1,0} broadcast(zero), dimensions={}, frontend_attributes={color="task_f"}
  broadcast.1 = f32[4,4]{1,0} broadcast(zero), dimensions={}, frontend_attributes={color="task_f"}
  ROOT tuple.1 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(broadcast.0, broadcast.1)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  Arg_1.2 = f32[4,4]{1,0} parameter(1), sharding={devices=[4,1]0,1,2,3}
  call.0 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(), to_apply=task_init, frontend_attributes={color="task_f"}
  get-tuple-element.4 = f32[4,4]{1,0} get-tuple-element(call.0), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  get-tuple-element.5 = f32[4,4]{1,0} get-tuple-element(call.0), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  call.2 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call.2), index=1, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(get-tuple-element.0, get-tuple-element.1, get-tuple-element.4, get-tuple-element.5), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.3), index=1, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.2, get-tuple-element.3)
}
)";

constexpr absl::string_view k4x1ShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 4
tile_assignment_dimensions: 1
iota_reshape_dims: 4
iota_transpose_perm: 0
)";

TEST_F(MpmdInsertReshardTest, BasicTasks) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kBasicTasksHlo, /*num_devices=*/4));

  TF_ASSERT_OK_AND_ASSIGN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 4}}, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 4, .num_devices = 4}}, {}));

  MpmdInsertReshard inserter{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, GetSharding(k4x1ShardingPbtxt));

  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(AllOf(op::CustomCall(std::string(kCustomCallReshard)),
                             op::Sharding(sharding), m::Color("task_g")))
                  .Times(4));
}

constexpr absl::string_view kCommonReshardHlo = R"(
task {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  param.1 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  add.0 = f32[4,4]{1,0} add(param.0, param.1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  add.1 = f32[4,4]{1,0} add(add.0, param.1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.0, add.1)
}

aggregate {
  param.2 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.3 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.4 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.5  = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.4 = f32[4,4]{1,0} add(param.2, param.3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.5 = f32[4,4]{1,0} add(param.4, param.5), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.4 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.4, add.5)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  Arg_1.2 = f32[4,4]{1,0} parameter(1), sharding={devices=[4,1]4,5,6,7}
  call.2 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task, frontend_attributes={color="task_g"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call.2), index=1, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2, get-tuple-element.0, get-tuple-element.1), to_apply=aggregate, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.3), index=1, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.2, get-tuple-element.3)
}
)";

TEST_F(MpmdInsertReshardTest, CommonReshard) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kCommonReshardHlo, /*num_devices=*/4));

  TF_ASSERT_OK_AND_ASSIGN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 4}}, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 4, .num_devices = 4}}, {}));

  MpmdInsertReshard inserter{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, GetSharding(k4x1ShardingPbtxt));

  // there should be a single reshard used in two places
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(AllOf(op::CustomCall(std::string(kCustomCallReshard)),
                             op::Sharding(sharding), m::Color("task_g"),
                             m::Users(ElementsAre(op::Call(), op::Call()))))
                  .Times(1));
}

constexpr absl::string_view kShardingChangeSameDevicesHlo = R"(
task_f {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}, sharding={replicated}
  add.0 = f32[4,4]{1,0} add(param.0, param.0), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.3 = (f32[4,4]{1,0}) tuple(add.0)
}

task_g {
  param.2 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.3 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.4 = f32[4,4]{1,0} add(param.2, param.3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.4 = (f32[4,4]{1,0}) tuple(add.4)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  call.2 = (f32[4,4]{1,0}) call(Arg_0.1), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(get-tuple-element.0, Arg_0.1), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.97 = (f32[4,4]{1,0}) tuple(get-tuple-element.2)
}
)";

TEST_F(MpmdInsertReshardTest, ShardingChangeSameDevices) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kShardingChangeSameDevicesHlo, /*num_devices=*/4));

  TF_ASSERT_OK_AND_ASSIGN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 4}}, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 4, .num_devices = 4}}, {}));

  MpmdInsertReshard inserter{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, GetSharding(k4x1ShardingPbtxt));

  // the parameter should get resharded into both uses
  // the first is over the same devices, but changes from sharded to replicated
  // the second has the same sharding, but different devices
  EXPECT_THAT(
      module->entry_computation()->parameter_instruction(0),
      m::Users(UnorderedElementsAre(
          AllOf(op::CustomCall(std::string(kCustomCallReshard)),
                op::Sharding(HloSharding::Replicate()), m::Color("task_f")),
          AllOf(op::CustomCall(std::string(kCustomCallReshard)),
                op::Sharding(sharding), m::Color("task_g")))));
}

constexpr absl::string_view kReshardOutputScalar = R"(
task_f {
  param.0 = f32[] parameter(0), frontend_attributes={color="task_f"}, sharding={replicated}
  add.0 = f32[] add(param.0, param.0), frontend_attributes={color="task_f"}, sharding={replicated}
  ROOT tuple.3 = (f32[]) tuple(add.0)
}
 
task_g {
  param.2 = f32[] parameter(0), frontend_attributes={color="task_g"}, sharding={replicated}
  param.3 = f32[] parameter(1), frontend_attributes={color="task_g"}, sharding={replicated}
  add.4 = f32[] add(param.2, param.3), frontend_attributes={color="task_g"}, sharding={replicated}
  ROOT tuple.4 = (f32[]) tuple(add.4)
}

ENTRY main {
  Arg_0.1 = f32[] parameter(0), sharding={replicated}
  call.2 = (f32[]) call(Arg_0.1), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[] get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}, sharding={replicated}
  call.3 = (f32[]) call(get-tuple-element.0, Arg_0.1), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[] get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}, sharding={replicated}
  ROOT tuple.97 = (f32[]) tuple(get-tuple-element.2)
}
)";

TEST_F(MpmdInsertReshardTest, ScalarReshard) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kReshardOutputScalar, /*num_devices=*/8));

  TF_ASSERT_OK_AND_ASSIGN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 4}}, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 4, .num_devices = 4}}, {}));

  MpmdInsertReshard inserter{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));

  // the parameter should get resharded into both uses
  // the first is over the same devices, but changes from sharded to replicated
  // the second has the same sharding, but different devices
  EXPECT_THAT(module->entry_computation()->root_instruction()->operands(),
              ElementsAre(AllOf(
                  op::CustomCall(std::string(kCustomCallReshard)),
                  m::Devices(*partition_, {{.start = 0, .num_devices = 8}}))));
}

constexpr absl::string_view kRootTupleRecolorHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[8]{0}, s32[])->(f32[], f32[8]{0})}, allow_spmd_sharding_propagation_to_parameters={true,false}, allow_spmd_sharding_propagation_to_output={false,true}

%f.1 (Arg_1.7: s32[], Arg_0.8: f32[8]) -> (f32[8]) {
  %Arg_0.8 = f32[8]{0} parameter(1), sharding={replicated}, frontend_attributes={color="f"}
  %multiply.32 = f32[8]{0} multiply(f32[8]{0} %Arg_0.8, f32[8]{0} %Arg_0.8), sharding={replicated}, frontend_attributes={color="f"}, metadata={op_name="jit(c)/jit(main)/jvp(jit(f.impl))/mul"}
  %Arg_1.7 = s32[] parameter(0), sharding={replicated}, frontend_attributes={color="f"}
  %convert.9 = f32[] convert(s32[] %Arg_1.7), sharding={replicated}, frontend_attributes={color="f"}, metadata={op_name="jit(c)/jit(main)/jvp(jit(f.impl))/convert_element_type[new_dtype=float32 weak_type=False sharding=None]"}
  %broadcast.16 = f32[8]{0} broadcast(f32[] %convert.9), dimensions={}, sharding={replicated}, frontend_attributes={color="f"}, metadata={op_name="jit(c)/jit(main)/jvp(jit(f.impl))/mul"}
  %multiply.33 = f32[8]{0} multiply(f32[8]{0} %multiply.32, f32[8]{0} %broadcast.16), sharding={replicated}, frontend_attributes={color="f"}, metadata={op_name="jit(c)/jit(main)/jvp(jit(f.impl))/mul"}
  ROOT %tuple.18 = (f32[8]{0}) tuple(f32[8]{0} %multiply.33)
}

%g.bwd.1 (multiply.34: f32[8]) -> (f32[8]) {
  %constant.12 = f32[] constant(1), sharding={replicated}
  %broadcast.18 = f32[8]{0} broadcast(f32[] %constant.12), dimensions={}, sharding={replicated}, frontend_attributes={color="g.bwd"}
  %negate.4 = f32[8]{0} negate(f32[8]{0} %broadcast.18), sharding={replicated}, frontend_attributes={color="g.bwd"}
  %multiply.34 = f32[8]{0} parameter(0), sharding={replicated}, frontend_attributes={color="g.bwd"}
  %sine.4 = f32[8]{0} sine(f32[8]{0} %multiply.34), sharding={replicated}, frontend_attributes={color="g.bwd"}
  %multiply.35 = f32[8]{0} multiply(f32[8]{0} %negate.4, f32[8]{0} %sine.4), sharding={replicated}, frontend_attributes={color="g.bwd"}
  ROOT %tuple.19 = (f32[8]{0}) tuple(f32[8]{0} %multiply.35)
}

%sharding.1 (add.15: f32[8]) -> (f32[8]) {
  %constant.13 = f32[] constant(1), sharding={replicated}
  %broadcast.20 = f32[8]{0} broadcast(f32[] %constant.13), dimensions={}, sharding={devices=[2]<=[2]}, frontend_attributes={color="g.bwd"}
  %add.15 = f32[8]{0} parameter(0), sharding={replicated}, frontend_attributes={color="sharding"}
  %add.16 = f32[8]{0} add(f32[8]{0} %broadcast.20, f32[8]{0} %add.15), sharding={devices=[2]<=[2]}, frontend_attributes={color="sharding"}
  ROOT %tuple.21 = (f32[8]{0}) tuple(f32[8]{0} %add.16)
}

%region_0.0 (Arg_0.5: f32[], Arg_1.4: f32[]) -> f32[] {
  %Arg_0.5 = f32[] parameter(0), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/reduce_sum[axes=(0,)]"}
  %Arg_1.4 = f32[] parameter(1), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/reduce_sum[axes=(0,)]"}
  ROOT %add.6 = f32[] add(f32[] %Arg_0.5, f32[] %Arg_1.4), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/reduce_sum[axes=(0,)]"}
}

%g.1 (multiply.40: f32[8], Arg_0.10: f32[8]) -> (f32[]) {
  %multiply.40 = f32[8]{0} parameter(0), sharding={replicated}, frontend_attributes={color="g"}
  %exp.4 = f32[8]{0} exponential(f32[8]{0} %multiply.40), sharding={replicated}, frontend_attributes={color="g"}, metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/cos"}
  %Arg_0.10 = f32[8]{0} parameter(1), sharding={replicated}, frontend_attributes={color="g"}
  %add.17 = f32[8]{0} add(f32[8]{0} %exp.4, f32[8]{0} %Arg_0.10), sharding={replicated}, frontend_attributes={color="g"}, metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/add"}
  %constant.14 = f32[] constant(0), frontend_attributes={color="g"}
  %reduce.4 = f32[] reduce(f32[8]{0} %add.17, f32[] %constant.14), dimensions={0}, to_apply=%region_0.0, sharding={replicated}, frontend_attributes={color="g"}, metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/reduce_sum[axes=(0,)]"}
  ROOT %tuple.22 = (f32[]) tuple(f32[] %reduce.4)
}

%f.bwd.3 (Arg_1.8: s32[], multiply.41: f32[8], Arg_0.11: f32[8]) -> (f32[8]) {
  %multiply.41 = f32[8]{0} parameter(1), sharding={replicated}, frontend_attributes={color="f.bwd"}
  %Arg_1.8 = s32[] parameter(0), sharding={replicated}, frontend_attributes={color="f.bwd"}
  %convert.10 = f32[] convert(s32[] %Arg_1.8), sharding={replicated}, frontend_attributes={color="f.bwd"}
  %broadcast.21 = f32[8]{0} broadcast(f32[] %convert.10), dimensions={}, sharding={replicated}, frontend_attributes={color="f.bwd"}
  %multiply.42 = f32[8]{0} multiply(f32[8]{0} %multiply.41, f32[8]{0} %broadcast.21), sharding={replicated}, frontend_attributes={color="f.bwd"}
  %Arg_0.11 = f32[8]{0} parameter(2), sharding={replicated}, frontend_attributes={color="f.bwd"}
  %multiply.43 = f32[8]{0} multiply(f32[8]{0} %multiply.42, f32[8]{0} %Arg_0.11), sharding={replicated}, frontend_attributes={color="f.bwd"}
  %multiply.44 = f32[8]{0} multiply(f32[8]{0} %Arg_0.11, f32[8]{0} %multiply.42), sharding={replicated}, frontend_attributes={color="f.bwd"}
  %add.18 = f32[8]{0} add(f32[8]{0} %multiply.43, f32[8]{0} %multiply.44), sharding={maximal device=0}, frontend_attributes={color="f.bwd"}
  ROOT %tuple.23 = (f32[8]{0}) tuple(f32[8]{0} %add.18)
}

ENTRY %main.100 (Arg_0.1: f32[8], Arg_1.2: s32[]) -> (f32[], f32[8]) {
  %Arg_1.2 = s32[] parameter(1), sharding={replicated}, frontend_attributes={color="sharding"}
  %Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}, frontend_attributes={color="f.bwd"}
  %call.15 = (f32[8]{0}) call(s32[] %Arg_1.2, f32[8]{0} %Arg_0.1), to_apply=%f.1, frontend_attributes={color="f"}
  %get-tuple-element.6 = f32[8]{0} get-tuple-element((f32[8]{0}) %call.15), index=0, sharding={replicated}, frontend_attributes={color="f"}
  %call.19 = (f32[]) call(f32[8]{0} %get-tuple-element.6, f32[8]{0} %Arg_0.1), to_apply=%g.1, frontend_attributes={color="g"}
  %get-tuple-element.10 = f32[] get-tuple-element((f32[]) %call.19), index=0, sharding={replicated}, frontend_attributes={color="g"}
  %custom-call = f32[] custom-call(f32[] %get-tuple-element.10), custom_call_target="RootTupleRecolor", sharding={replicated}, frontend_attributes={color="sharding"}
  %call.16 = (f32[8]{0}) call(f32[8]{0} %get-tuple-element.6), to_apply=%g.bwd.1, frontend_attributes={color="g.bwd"}
  %get-tuple-element.7 = f32[8]{0} get-tuple-element((f32[8]{0}) %call.16), index=0, sharding={replicated}, frontend_attributes={color="g.bwd"}
  %call.20 = (f32[8]{0}) call(s32[] %Arg_1.2, f32[8]{0} %get-tuple-element.7, f32[8]{0} %Arg_0.1), to_apply=%f.bwd.3, frontend_attributes={color="f.bwd"}
  %get-tuple-element.11 = f32[8]{0} get-tuple-element((f32[8]{0}) %call.20), index=0, sharding={maximal device=0}, frontend_attributes={color="f.bwd"}
  %call.18 = (f32[8]{0}) call(f32[8]{0} %get-tuple-element.11), to_apply=%sharding.1, frontend_attributes={color="sharding"}
  %get-tuple-element.9 = f32[8]{0} get-tuple-element((f32[8]{0}) %call.18), index=0, sharding={devices=[2]<=[2]}, frontend_attributes={color="sharding"}
  ROOT %tuple.99 = (f32[], f32[8]{0}) tuple(f32[] %custom-call, f32[8]{0} %get-tuple-element.9), sharding={{replicated}, {devices=[2]<=[2]}}, frontend_attributes={color="sharding"}
}
)";

TEST_F(MpmdInsertReshardTest, RootTupleRecolor) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kRootTupleRecolorHlo, /*num_devices=*/2));

  TF_ASSERT_OK_AND_ASSIGN(
      auto f, partition_->AllocateColor(
                  "f", zuku::DeviceList{{.start = 0, .num_devices = 1}}, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto g, partition_->AllocateColor(
                  "g", zuku::DeviceList{{.start = 1, .num_devices = 1}}, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto f_bwd,
      partition_->AllocateColor(
          "f.bwd", zuku::DeviceList{{.start = 0, .num_devices = 1}}, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto g_bwd,
      partition_->AllocateColor(
          "g.bwd", zuku::DeviceList{{.start = 1, .num_devices = 1}}, {}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto global,
      partition_->AllocateColor(
          "sharding", zuku::DeviceList{{.start = 0, .num_devices = 2}}, {}));

  MpmdInsertReshard inserter{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));

  EXPECT_THAT(
      module->entry_computation()->root_instruction()->operands(),
      AllOf(Contains(AllOf(
                op::CustomCall(std::string(kCustomCallReshard)),
                m::Devices(*partition_, {{.start = 0, .num_devices = 2}}))),
            Not(Contains(
                op::CustomCall(std::string(kCustomCallRootTupleRecolor))))));
}

}  // namespace
}  // namespace xla
