/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_repack_optimization_barrier.h"

#include "gmock/gmock.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdRepackOptimizationBarrierTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

static constexpr absl::string_view kTaskWithBarrierHloSingle = R"(
region_0.10 {
  %Arg_0.11 = f32[] parameter(0)
  %Arg_1.12 = f32[] parameter(1)
  ROOT %add.13 = f32[] add(f32[] %Arg_0.11, f32[] %Arg_1.12), metadata={op_name="jit(c)/jit(main)/task_g/reduce_sum[axes=(0,)]"}
}

ENTRY %main.15 {
  %Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  %multiply.4 = f32[8]{0} multiply(f32[8]{0} %Arg_0.1, f32[8]{0} %Arg_0.1), metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  %Arg_1.2 = s32[] parameter(1), sharding={replicated}
  %convert.5 = f32[] convert(s32[] %Arg_1.2), metadata={op_name="jit(c)/jit(main)/task_f/convert_element_type[new_dtype=float32 weak_type=False]"}
  %broadcast.6 = f32[8]{0} broadcast(f32[] %convert.5), dimensions={}, metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  %multiply.7 = f32[8]{0} multiply(f32[8]{0} %multiply.4, f32[8]{0} %broadcast.6), metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  %tangent.8 = f32[8]{0} tan(f32[8]{0} %multiply.7), metadata={op_name="jit(c)/jit(main)/task_g/tan"}
  %custom-call = f32[8]{0} custom-call(f32[8]{0} %tangent.8), custom_call_target="UnpackedOptimizationBarrier", backend_config="opt-barrier.1"
  %add.9 = f32[8]{0} add(f32[8]{0} %custom-call, f32[8]{0} %multiply.7), metadata={op_name="jit(c)/jit(main)/task_g/add"}
  ROOT %add.10 = f32[] add(f32[8]{0} %custom-call, f32[8]{0} %add.9)
} // main.15
)";

TEST_F(MpmdRepackOptimizationBarrierTest, TaskWithBarrierSingle) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetHloModuleFromText(kTaskWithBarrierHloSingle,
                                               /*num_devices=*/4));

  MpmdRepackOptimizationBarrier repacker{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, repacker.Run(module.get()));

  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(op::OptimizationBarrier(op::Tan())));
}

static constexpr absl::string_view kTaskWithBarrierHloTuple = R"(
region_0.10 {
  %Arg_0.11 = f32[] parameter(0)
  %Arg_1.12 = f32[] parameter(1)
  ROOT %add.13 = f32[] add(f32[] %Arg_0.11, f32[] %Arg_1.12), metadata={op_name="jit(c)/jit(main)/task_g/reduce_sum[axes=(0,)]"}
}

ENTRY %main.15 {
  %Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  %multiply.4 = f32[8]{0} multiply(f32[8]{0} %Arg_0.1, f32[8]{0} %Arg_0.1), metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  %Arg_1.2 = s32[] parameter(1), sharding={replicated}
  %convert.5 = f32[] convert(s32[] %Arg_1.2), metadata={op_name="jit(c)/jit(main)/task_f/convert_element_type[new_dtype=float32 weak_type=False]"}
  %broadcast.6 = f32[8]{0} broadcast(f32[] %convert.5), dimensions={}, metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  %multiply.7 = f32[8]{0} multiply(f32[8]{0} %multiply.4, f32[8]{0} %broadcast.6), metadata={op_name="jit(c)/jit(main)/task_f/mul"}
  %tangent.8 = f32[8]{0} tan(f32[8]{0} %multiply.7), metadata={op_name="jit(c)/jit(main)/task_g/tan"}
  %custom-call = f32[8]{0} custom-call(f32[8]{0} %multiply.7), custom_call_target="UnpackedOptimizationBarrier", backend_config="opt-barrier.1"
  %custom-call.1 = f32[8]{0} custom-call(f32[8]{0} %tangent.8), custom_call_target="UnpackedOptimizationBarrier", backend_config="opt-barrier.1"
  %add.9 = f32[8]{0} add(f32[8]{0} %custom-call.1, f32[8]{0} %multiply.7), metadata={op_name="jit(c)/jit(main)/task_g/add"}
  %custom-call.2 = f32[8]{0} custom-call(f32[8]{0} %add.9), custom_call_target="UnpackedOptimizationBarrier", backend_config="opt-barrier.2"
  %custom-call.3 = f32[8]{0} custom-call(f32[8]{0} %tangent.8), custom_call_target="UnpackedOptimizationBarrier", backend_config="opt-barrier.2"
  %add.10 = f32[8]{0} add(f32[8]{0} %custom-call.1, f32[8]{0} %custom-call.2)
  ROOT %tuple = (f32[8]{0}, f32[8]{0}, f32[8]{0}) tuple(f32[8]{0} %custom-call, f32[8]{0} %custom-call.3, f32[8]{0} %add.10)
}
)";

TEST_F(MpmdRepackOptimizationBarrierTest, TaskWithBarrierTuple) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetHloModuleFromText(kTaskWithBarrierHloTuple,
                                               /*num_devices=*/4));

  MpmdRepackOptimizationBarrier repacker{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, repacker.Run(module.get()));

  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(op::OptimizationBarrier()).Times(2));
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(op::Tuple(op::Multiply(), op::Tan())));
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(op::Tuple(op::Add(), op::Tan())));
}

}  // namespace
}  // namespace xla
