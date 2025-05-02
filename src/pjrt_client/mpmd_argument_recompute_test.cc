/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_argument_recompute.h"

#include <utility>

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class MpmdArgumentRecomputeTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

static constexpr absl::string_view kBasicRecomputeHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[1024]{0}, f32[1024]{0}, f32[1024]{0})->(f32[1024]{0}, f32[1024]{0})}

ENTRY main.117 {
  Arg_0.1 = f32[1024]{0} parameter(0), frontend_attributes={color="red"}
  Arg_1.2 = f32[1024]{0} parameter(1), frontend_attributes={color="red"}
  Arg_2.3 = f32[1024,1024]{1,0} parameter(2), frontend_attributes={color="blue"}
  Arg_3.4 = f32[1024]{0} parameter(3), frontend_attributes={color="blue"}
  exp.0 = f32[1024]{0} exponential(Arg_0.1), frontend_attributes={color="red"}
  add.1 = f32[1024]{0} add(Arg_1.2, Arg_0.1), frontend_attributes={color="red"}
  add.2 = f32[1024]{0} add(exp.0, exp.0), frontend_attributes={color="red"}
  broadcast = f32[1024,1024]{1,0} broadcast(add.2), dimensions={0}, frontend_attributes={color="red"}
  dot.0 = f32[1024]{0} dot(broadcast, Arg_2.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={color="blue"}
  add.3 = f32[1024]{0} add(dot.0, Arg_3.4), frontend_attributes={color="blue"}
  ROOT tuple = (f32[1024], f32[1024]) tuple(add.1, add.3)
}
)";

TEST_F(MpmdArgumentRecomputeTest, BasicRecomputeFromArguments) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kBasicRecomputeHlo, /*num_devices=*/2));
  TF_ASSIGN_OR_RETURN(auto red,
                      partition_->AllocateColor(
                          "red", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto blue,
                      partition_->AllocateColor(
                          "blue", {{.start = 0, .num_devices = 2}}, nullptr));
  // no rec
  {
    // set a low recompute cost and make sure nothing is recomputed
    MpmdArgumentRecompute recompute{partition_.get(),
                                    /*max_recompute_cost=*/1,
                                    /*global_mem_to_compute_ratio=*/0};

    TF_ASSERT_OK_AND_ASSIGN(bool changed, recompute.Run(module.get()));
    EXPECT_FALSE(changed);
  }

  MpmdArgumentRecompute recompute{partition_.get(),
                                  /*max_recompute_cost=*/4096,
                                  /*global_mem_to_compute_ratio=*/10000};

  TF_ASSERT_OK_AND_ASSIGN(bool changed, recompute.Run(module.get()));

  // after recomputing, the red exp should be recognized
  // as not having any more users and should be removed
  EXPECT_THAT(module->entry_computation()->instructions(),
              AllOf(Contains(AllOf(op::Exp(), m::Color("blue"))).Times(1),
                    Contains(AllOf(op::Broadcast(), m::Color("blue"))).Times(1),
                    Not(Contains(AllOf(op::Exp(), m::Color("red"))))));
}

static constexpr absl::string_view kRecomputeSharedCloneHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[1024]{0}, f32[1024]{0}, f32[1024]{0})->(f32[1024]{0}, f32[1024]{0})}

ENTRY main.117 {
  Arg_0.1 = f32[1024]{0} parameter(0), frontend_attributes={color="red"}
  Arg_1.2 = f32[1024]{0} parameter(1), frontend_attributes={color="red"}
  Arg_2.3 = f32[1024,1024]{1,0} parameter(2), frontend_attributes={color="blue"}
  Arg_3.4 = f32[1024]{0} parameter(3), frontend_attributes={color="blue"}
  exp.0 = f32[1024]{0} exponential(Arg_0.1), frontend_attributes={color="red"}
  add.1 = f32[1024]{0} add(Arg_1.2, Arg_0.1), frontend_attributes={color="red"}
  add.2 = f32[1024]{0} add(exp.0, exp.0), frontend_attributes={color="red"}
  multiply.0 = f32[1024]{0} multiply(add.2, add.2), frontend_attributes={color="red"}
  multiply.1 = f32[1024]{0} multiply(add.2, add.2), frontend_attributes={color="red"}
  broadcast.0 = f32[1024,1024]{1,0} broadcast(multiply.0), dimensions={0}, frontend_attributes={color="red"}
  broadcast.1 = f32[1024,1024]{1,0} broadcast(multiply.1), dimensions={0}, frontend_attributes={color="red"}
  dot.0 = f32[1024]{0} dot(broadcast.0, Arg_2.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={color="blue"}
  add.3 = f32[1024]{0} add(dot.0, Arg_3.4), frontend_attributes={color="blue"}
  dot.1 = f32[1024]{0} dot(broadcast.1, Arg_2.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={color="blue"}
  add.4 = f32[1024]{0} add(dot.1, Arg_3.4), frontend_attributes={color="blue"}
  ROOT tuple = (f32[1024], f32[1024], f32[1024]) tuple(add.1, add.3, add.4)
}
)";

TEST_F(MpmdArgumentRecomputeTest, RecomputeSharedClone) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kRecomputeSharedCloneHlo, /*num_devices=*/2));
  TF_ASSIGN_OR_RETURN(auto red,
                      partition_->AllocateColor(
                          "red", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto blue,
                      partition_->AllocateColor(
                          "blue", {{.start = 0, .num_devices = 2}}, nullptr));
  MpmdArgumentRecompute recompute{partition_.get(),
                                  /*max_recompute_cost=*/8192};

  TF_ASSERT_OK_AND_ASSIGN(bool changed, recompute.Run(module.get()));

  // after recomputing, the red exp should be recognized
  // as not having any more users and should be removed
  EXPECT_THAT(module->entry_computation()->instructions(),
              AllOf(Contains(AllOf(op::Exp(), m::Color("blue"))).Times(1),
                    Contains(AllOf(op::Add(), m::Color("blue"))).Times(3),
                    Contains(AllOf(op::Broadcast(), m::Color("blue"))).Times(2),
                    Contains(AllOf(op::Add(), m::Color("red"))).Times(1),
                    Not(Contains(AllOf(op::Broadcast(), m::Color("red")))),
                    Not(Contains(AllOf(op::Exp(), m::Color("red"))))));
}

static constexpr absl::string_view kReplicatedArgumentHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[1024]{0}, f32[1024]{0}, f32[1024]{0})->(f32[1024]{0}, f32[1024]{0})}

ENTRY main.117 {
  Arg_0.1 = f32[1024]{0} parameter(0), frontend_attributes={color="red"}, sharding={devices=[2]<=[2]}
  Arg_1.2 = f32[512]{0} parameter(1), frontend_attributes={color="red"}, sharding={devices=[2]<=[2]}
  Arg_2.3 = f32[1024]{0} parameter(2), frontend_attributes={color="blue"}, sharding={devices=[2]<=[2]}
  replicated-Arg_0.1 = f32[512]{0} copy(Arg_0.1), frontend_attributes={color="red"}, sharding={replicated}
  constant.0 = s32[] constant(0)
  slice = f32[512]{0} dynamic-slice(replicated-Arg_0.1, constant.0), dynamic_slice_sizes={512}, frontend_attributes={color="red"}
  exp.0 = f32[512]{0} exponential(slice), frontend_attributes={color="red"}
  add.1 = f32[512]{0} add(slice, Arg_1.2), frontend_attributes={color="red"}
  add.2 = f32[512]{0} add(exp.0, exp.0), frontend_attributes={color="red"}
  multiply.0 = f32[512]{0} multiply(add.2, add.2), frontend_attributes={color="red"}
  multiply.1 = f32[512]{0} multiply(add.2, add.2), frontend_attributes={color="red"}
  broadcast.0 = f32[512,1024]{1,0} broadcast(multiply.0), dimensions={1}, frontend_attributes={color="red"}
  broadcast.1 = f32[1024,512]{1,0} broadcast(multiply.1), dimensions={0}, frontend_attributes={color="red"}
  dot.0 = f32[1024]{0} dot(broadcast.0, broadcast.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, frontend_attributes={color="blue"}
  add.3 = f32[1024]{0} add(dot.0, Arg_2.3), frontend_attributes={color="blue"}
  ROOT tuple = (f32[512], f32[1024]) tuple(add.1, add.3)
}
)";

TEST_F(MpmdArgumentRecomputeTest, RecomputeFromReplicatedArgument) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kReplicatedArgumentHlo, /*num_devices=*/2));

  TF_ASSIGN_OR_RETURN(auto red,
                      partition_->AllocateColor(
                          "red", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto blue,
                      partition_->AllocateColor(
                          "blue", {{.start = 0, .num_devices = 2}}, nullptr));
  MpmdArgumentRecompute recompute{partition_.get(),
                                  /*max_recompute_cost=*/4096};

  TF_ASSERT_OK_AND_ASSIGN(bool changed, recompute.Run(module.get()));

  // after recomputing, the red exp should be recognized
  // as not having any more users and should be removed
  EXPECT_THAT(
      module->entry_computation()->instructions(),
      AllOf(Contains(AllOf(op::Exp(), m::Color("blue"))).Times(1),
            Contains(AllOf(op::Broadcast(), m::Color("blue"))).Times(2),
            Contains(AllOf(op::DynamicSlice(), m::Color("blue"))).Times(1),
            Not(Contains(AllOf(op::Copy(), m::Color("blue")))),
            Contains(AllOf(op::DynamicSlice(), m::Color("red"))).Times(1),
            Contains(AllOf(op::Copy(), m::Color("red"))).Times(1),
            Not(Contains(AllOf(op::Broadcast(), m::Color("red")))),
            Not(Contains(AllOf(op::Exp(), m::Color("red"))))));
}

TEST_F(MpmdArgumentRecomputeTest, DynamicSliceWrongSizeSharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("wrong-sharding-size-argument-recompute.txt",
                           /*num_devices=*/8));

  zuku::DeviceList devices04{{.start = 0, .num_devices = 4}};
  zuku::DeviceList devices48{{.start = 4, .num_devices = 4}};
  zuku::DeviceList devices08{{.start = 0, .num_devices = 8}};
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

  TF_ASSIGN_OR_RETURN(auto li, partition_->AllocateLoopIncrementColor());
  TF_ASSIGN_OR_RETURN(
      auto rp, partition_->AllocateColor("replicated-params", devices08));
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

  MpmdArgumentRecompute recompute{partition_.get(), 1024 * 1024};
  TF_ASSIGN_OR_RETURN(bool changed, recompute.Run(module.get()));
  TF_ASSERT_OK(VerifyHloModule(*module, *partition_));
}

}  // namespace
}  // namespace xla
