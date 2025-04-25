/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_unpack_optimization_barrier.h"

#include "gmock/gmock.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdUnpackOptimizationBarrierTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

static constexpr absl::string_view kTaskWithBarrierHloSingle = R"(
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
  opt-barrier.1 = f32[8]{0} opt-barrier(cosine.8)
  add.9 = f32[8]{0} add(opt-barrier.1, multiply.7), metadata={op_name="jit(c)/jit(main)/task_g/add"}
  ROOT add.10 = f32[] add(opt-barrier.1, add.9)
} // main.15
)";

TEST_F(MpmdUnpackOptimizationBarrierTest, TaskWithBarrierSingle) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetHloModuleFromText(kTaskWithBarrierHloSingle,
                                               /*num_devices=*/4));

  MpmdUnpackOptimizationBarrier unpacker{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, unpacker.Run(module.get()));

  std::cerr << module->ToString() << std::endl;
}

static constexpr absl::string_view kTaskWithBarrierHloTuple = R"(
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
  tuple.1 = (f32[8]{0}, f32[8]{0}) tuple(multiply.7, cosine.8)
  opt-barrier.1 = (f32[8]{0}, f32[8]{0}) opt-barrier(tuple.1)
  gte.cosine.8 = get-tuple-element(opt-barrier.1), index=1
  add.9 = f32[8]{0} add(gte.cosine.8, multiply.7), metadata={op_name="jit(c)/jit(main)/task_g/add"}
  tuple.2 = (f32[8]{0}, f32[8]{0}) tuple(add.9, cosine.8)
  opt-barrier.2 = (f32[8]{0}, f32[8]{0}) opt-barrier(tuple.2)
  gte.add.9 = get-tuple-element(opt-barrier.2), index=0
  ROOT add.10 = f32[] add(gte.cosine.8, gte.add.9)
} // main.15
)";

TEST_F(MpmdUnpackOptimizationBarrierTest, TaskWithBarrierTuple) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetHloModuleFromText(kTaskWithBarrierHloTuple,
                                               /*num_devices=*/4));

  MpmdUnpackOptimizationBarrier unpacker{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, unpacker.Run(module.get()));

  std::cerr << module->ToString() << std::endl;
}

}  // namespace
}  // namespace xla
