/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_logical_sharding_propagation.h"

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdLogicalShardingPropagationTest : public MpmdTestBase {};

using ::testing::ElementsAre;

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

constexpr absl::string_view kSimpleAutoshardedHlo = R"(
region_0.10 {
  Arg_0.11 = f32[] parameter(0)
  Arg_1.12 = f32[] parameter(1)
  ROOT add.13 = f32[] add(Arg_0.11, Arg_1.12), metadata={op_name="jit(c)/jit(main)/task_g/reduce_sum[axes=(0,)]"}
}

ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  custom-call = f32[8]{0} custom-call(Arg_0.1), custom_call_target="AutoSharding", backend_config={"axes": [["x"]]}
  multiply.4 = f32[8]{0} multiply(custom-call, custom-call)
  multiply.7 = f32[8]{0} multiply(multiply.4, multiply.4)
  cosine.8 = f32[8]{0} cosine(multiply.7)
  custom-call.1 = f32[8]{0} custom-call(cosine.8), custom_call_target="AutoSharding", backend_config={"axes": [["x"]]}
  add.9 = f32[8]{0} add(cosine.8, multiply.7)
  constant.3 = f32[] constant(0)
  broadcast.3 = f32[8]{0} broadcast(constant.3), dimensions={}
  ROOT add.14 = f32[] add(add.9, broadcast.3)
} // main.15
)";

TEST_F(MpmdLogicalShardingPropagationTest, SimpleLogicalSharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kSimpleAutoshardedHlo, /*num_devices=*/4));

  MpmdLogicalShardingPropagation prop;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prop.Run(module.get()));

  // the autosharding custom calls should have been removed
  EXPECT_THAT(module->entry_computation()->instructions(),
              Each(Not(op::CustomCall("AutoSharding"))));

  // all that one non-trivial math ops should have autosharding assigned
  EXPECT_THAT(
      module->entry_computation()->instructions(),
      AllOf(Contains(AllOf(m::NontrivialOp(), m::HasLogicalAxes())).Times(2),
            Contains(AllOf(op::Parameter(), m::HasLogicalAxes())).Times(1)));
}

constexpr absl::string_view kDotLogicalSharding = R"(
ENTRY main.15 {
  Arg_0.1 = f32[8,2048,12288]{2,1,0} parameter(0)
  custom-call.0 = f32[8,2048,12288]{2,1,0} custom-call(Arg_0.1), custom_call_target="AutoSharding", backend_config={"axes": [["data", "replica"], ["seq"], ["mdl"]]}
  Arg_1.2 = f32[8,12288,2048]{2,1,0} parameter(1)
  custom-call.1 = f32[8,12288,2048]{2,1,0} custom-call(Arg_1.2), custom_call_target="AutoSharding", backend_config={"axes": [["data", "replica"], ["hidden"], ["seq"]]}
  ROOT dot.0 = f32[12288,12288]{1,0} dot(custom-call.0, custom-call.1), lhs_contracting_dims={0,1}, rhs_contracting_dims={0,2}
} // main.15
)";

TEST_F(MpmdLogicalShardingPropagationTest, DotLogicalSharding) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetHloModuleFromText(kDotLogicalSharding,
                                                            /*num_devices=*/4));

  MpmdLogicalShardingPropagation prop;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prop.Run(module.get()));

  auto axes = GetAxes(module->entry_computation()->root_instruction());
  ASSERT_TRUE(axes.has_value());
  EXPECT_THAT(axes->axes,
              ElementsAre(ElementsAre("mdl"), ElementsAre("hidden")));
}

constexpr absl::string_view kBatchDimDot = R"(
ENTRY main {
  concatenate.65 = bf16[2,256,16,128]{3,2,1,0} parameter(0), frontend_attributes={axes={"axes": [["data", "fsdp"], [], ["tensor", "sequence"], []]}}
  concatenate.64 = bf16[2,256,16,128]{3,2,1,0} parameter(1), frontend_attributes={axes={"axes": [["data", "fsdp"], [], ["tensor", "sequence"], []]}}
  dot.495 = bf16[2,16,256,256]{3,2,1,0} dot(concatenate.64, concatenate.65), lhs_batch_dims={0,2}, lhs_contracting_dims={3}, rhs_batch_dims={0,2}, rhs_contracting_dims={3}
}
)";
TEST_F(MpmdLogicalShardingPropagationTest, BatchDimDotLogicalSharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kBatchDimDot, /*num_devices=*/4));

  MpmdLogicalShardingPropagation prop;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, prop.Run(module.get()));

  auto axes = GetAxes(module->entry_computation()->root_instruction());
  ASSERT_TRUE(axes.has_value());
  EXPECT_THAT(axes->axes, ElementsAre(ElementsAre("data", "fsdp"),
                                      ElementsAre("tensor", "sequence"),
                                      ElementsAre(), ElementsAre()));
}

}  // namespace
}  // namespace xla
