/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_hoist_loop_convert.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::AllOf;

class MpmdHoistLoopConvertTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;

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

TEST_F(MpmdHoistLoopConvertTest, BasicLoop) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kMultipleMicrobatchSlicesHlo, /*num_devices=*/1));

  GTEST_SKIP() << "Input module needs to be flattened";

  MpmdHoistLoopConvert hoister{partition_.get()};
  TF_ASSERT_OK(hoister.Run(module.get()).status());

  Shape bf16_4x4{PrimitiveType::BF16, {4, 4}, {}, {}};
  bf16_4x4.mutable_layout()->add_minor_to_major(1);
  bf16_4x4.mutable_layout()->add_minor_to_major(0);

  // 3 converts and copies of the convert should have been hoisted out of the
  // loop
  EXPECT_THAT(
      module->entry_computation()->instructions(),
      AllOf(Contains(AllOf(op::Convert(), op::Shape(bf16_4x4))).Times(3),
            Contains(AllOf(op::Copy(), op::Shape(bf16_4x4))).Times(3)));
}

TEST_F(MpmdHoistLoopConvertTest, FailedHoist) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("failed_hoist_convert.txt", /*num_devices=*/8));

  MpmdHoistLoopConvert hoister{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));
}

}  // namespace
}  // namespace xla
