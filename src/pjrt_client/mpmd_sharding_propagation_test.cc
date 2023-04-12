/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_sharding_propagation.h"

#include <utility>

#include "gmock/gmock.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/mpmd_coloring.h"
#include "xla/pjrt/multimesh/mpmd_computation_grouper.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdShardingPropagationTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

static constexpr absl::string_view kSimpleShardedHlo = R"(
ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={devices=[2]<=[2]}
  multiply.4 = f32[8]{0} multiply(Arg_0.1, Arg_0.1), metadata={op_name="task_f"}
  multiply.7 = f32[8]{0} multiply(multiply.4, multiply.4), metadata={op_name="task_f"}
  cosine.8 = f32[8]{0} cosine(multiply.7)
  add.9 = f32[8]{0} add(cosine.8, multiply.7), metadata={op_name="task_g"}
  constant.3 = f32[] constant(0)
  broadcast.3 = f32[8]{0} broadcast(constant.3), dimensions={}
  ROOT add.14 = f32[] add(add.9, broadcast.3)
} // main.15
)";

TEST_F(MpmdShardingPropagationTest, SimpleAutosharding) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kSimpleShardedHlo, /*num_devices=*/4));

  RegisterNamedTestTask("task_f", {0, 2}, {2}, {"x"}, {{"batch", "x"}});
  RegisterNamedTestTask("task_g", {2, 4}, {2}, {"x"}, {{"batch", "x"}});

  HloSharding correct_sharding =
      module->entry_computation()->parameter_instruction(0)->sharding();

  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(changed, grouper.Run(module.get()));

  MpmdShardingPropagation propagation{
      partition_.get(), MpmdShardingPropagation::PropagationMode::ForwardFull};
  TF_ASSERT_OK_AND_ASSIGN(changed, propagation.Run(module.get()));

  for (auto* instruction : module->entry_computation()->instructions()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      EXPECT_THAT(
          instruction->called_computations()[0]->instructions(),
          Each(AnyOf(AllOf(m::NontrivialOp(), op::Sharding(correct_sharding)),
                     m::TrivialOp())));
    }
  }
}

static constexpr absl::string_view kNestedWhileHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4]{0}, f32[4]{0})->f32[]}, num_partitions=2

%task_f_loop (get-tuple-element.4: f32[4,4], get-tuple-element.5: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.4 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %get-tuple-element.5 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %add.0 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.4, f32[4,4]{1,0} %get-tuple-element.5), frontend_attributes={color="task_f"}
  %add.1 = f32[4,4]{1,0} add(f32[4,4]{1,0} %add.0, f32[4,4]{1,0} %get-tuple-element.5), frontend_attributes={color="task_f"}
  ROOT %tuple.3 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.1)
}

%task_g_loop (get-tuple-element.7: f32[4,4], get-tuple-element.8: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.7 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}
  %get-tuple-element.8 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}
  %add.4 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.7, f32[4,4]{1,0} %get-tuple-element.8), frontend_attributes={color="task_g"}
  %add.5 = f32[4,4]{1,0} add(f32[4,4]{1,0} %add.4, f32[4,4]{1,0} %get-tuple-element.8), frontend_attributes={color="task_g"}
  ROOT %tuple.4 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.5)
}

%region_0.83 (arg_tuple.84: (f32[4,4], f32[4,4], f32[4,4], f32[4,4])) -> (f32[4,4], f32[4,4], f32[4,4], f32[4,4]) {
  %arg_tuple.84 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  %get-tuple-element.10 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=0
  %get-tuple-element.11 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=1
  %get-tuple-element.12 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=2
  %call.2 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.10, f32[4,4]{1,0} %get-tuple-element.12), to_apply=%task_f_loop, frontend_attributes={color="task_f"}
  %get-tuple-element.6 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.2), index=0, frontend_attributes={color="task_f"}
  %get-tuple-element.13 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=3
  %call.3 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.11, f32[4,4]{1,0} %get-tuple-element.13), to_apply=%task_g_loop, frontend_attributes={color="task_g"}
  %get-tuple-element.9 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.3), index=0, frontend_attributes={color="task_g"}
  ROOT %tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.10, f32[4,4]{1,0} %get-tuple-element.11, f32[4,4]{1,0} %get-tuple-element.6, f32[4,4]{1,0} %get-tuple-element.9)
}

%region_2.98 (arg_tuple.99: (f32[4,4], f32[4,4], f32[4,4], f32[4,4])) -> pred[] {
  %arg_tuple.99 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  %constant.106 = s32[] constant(2)
  %constant.107 = s32[] constant(2)
  ROOT %compare.108 = pred[] compare(s32[] %constant.106, s32[] %constant.107), direction=LT
}

%task_f () -> (f32[4,4]) {
  %constant.0 = f32[] constant(0), frontend_attributes={color="task_f"}
  %broadcast.1 = f32[4,4]{1,0} broadcast(f32[] %constant.0), dimensions={}, frontend_attributes={color="task_f"}
  %copy.0 = f32[4,4]{1,0} copy(f32[4,4]{1,0} %broadcast.1), frontend_attributes={color="task_f"}
  ROOT %tuple.1 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %copy.0)
}

%task_g () -> (f32[4,4]) {
  %constant.1 = f32[] constant(0), frontend_attributes={color="task_g"}
  %broadcast.2 = f32[4,4]{1,0} broadcast(f32[] %constant.1), dimensions={}, frontend_attributes={color="task_g"}
  %copy.1 = f32[4,4]{1,0} copy(f32[4,4]{1,0} %broadcast.2), frontend_attributes={color="task_g"}
  ROOT %tuple.2 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %copy.1)
}

%task_f.1 (get-tuple-element.14: f32[4,4], Arg_2.0: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.14 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %Arg_2.0 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %add.6 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.14, f32[4,4]{1,0} %Arg_2.0), frontend_attributes={color="task_f"}
  ROOT %tuple.5 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.6)
}

%task_g.1 (get-tuple-element.16: f32[4,4], Arg_3.0: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.16 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}
  %Arg_3.0 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}
  %add.7 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.16, f32[4,4]{1,0} %Arg_3.0), frontend_attributes={color="task_g"}
  ROOT %tuple.6 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.7)
}

ENTRY %main.117 (Arg_0.1: f32[4,4], Arg_1.2: f32[4,4], Arg_2.3: f32[4,4], Arg_3.4: f32[4,4]) -> (f32[4,4], f32[4,4]) {
  %Arg_0.1 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}, sharding={devices=[2,1]<=[2]}
  %Arg_1.2 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}, sharding={devices=[2,1]<=[2]}
  %call = (f32[4,4]{1,0}) call(), to_apply=%task_f, frontend_attributes={color="task_f"}
  %get-tuple-element = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call), index=0, frontend_attributes={color="task_f"}
  %call.1 = (f32[4,4]{1,0}) call(), to_apply=%task_g, frontend_attributes={color="task_g"}
  %get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.1), index=0, frontend_attributes={color="task_g"}
  %tuple.10 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %Arg_0.1, f32[4,4]{1,0} %Arg_1.2, f32[4,4]{1,0} %get-tuple-element, f32[4,4]{1,0} %get-tuple-element.1)
  %while.109 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) while((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.10), condition=%region_2.98, body=%region_0.83, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  %get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.109), index=2
  %Arg_2.3 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_f"}
  %call.4 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.2, f32[4,4]{1,0} %Arg_2.3), to_apply=%task_f.1, frontend_attributes={color="task_f"}
  %get-tuple-element.15 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.4), index=0, frontend_attributes={color="task_f"}
  %get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.109), index=3
  %Arg_3.4 = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_g"}
  %call.5 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.3, f32[4,4]{1,0} %Arg_3.4), to_apply=%task_g.1, frontend_attributes={color="task_g"}
  %get-tuple-element.17 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.5), index=0, frontend_attributes={color="task_g"}
  ROOT %tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.15, f32[4,4]{1,0} %get-tuple-element.17)
}
)";

static constexpr absl::string_view kSharding4xPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 4
tile_assignment_dimensions: 1
iota_reshape_dims: 4
iota_transpose_perm: 0
)";

TEST_F(MpmdShardingPropagationTest, NestedWhileLoop) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kNestedWhileHlo, /*num_devices=*/4));

  TF_ASSERT_OK_AND_ASSIGN(
      auto f, partition_->AllocateColor(
                  "task_f", {{.start = 0, .num_devices = 4}}, nullptr));
  TF_ASSERT_OK_AND_ASSIGN(
      auto g, partition_->AllocateColor(
                  "task_g", {{.start = 0, .num_devices = 4}}, nullptr));
  MpmdShardingPropagation propagation{
      partition_.get(), MpmdShardingPropagation::PropagationMode::ForwardFull};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, propagation.Run(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding4x,
                          GetSharding(kSharding4xPbtxt));

  // every add and copy operation should have been assigned a non-replicated
  // sharding
  EXPECT_THAT(m::FlatInstructionsWithoutReducesAndPredicates(module.get()),
              Each(Not(AllOf(AnyOf(op::Add(), op::Copy()),
                             Not(op::Sharding(sharding4x))))));
}

TEST_F(MpmdShardingPropagationTest, ShardReplicatedIntermediates) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromPath("large_replicated_intermediate.txt",
                                        /*num_devices=*/8));
  TF_ASSIGN_OR_RETURN(auto blue,
                      partition_->AllocateColor(
                          "blue", {{.start = 0, .num_devices = 8}}, nullptr));
  MpmdShardingPropagation propagation{
      partition_.get(),
      MpmdShardingPropagation::PropagationMode::BackwardInputOutput};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, propagation.Run(module.get()));
}

TEST_F(MpmdShardingPropagationTest, LargeParametersNotSharded) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("propagate_large_parameters_not_sharded.txt",
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

  TF_ASSIGN_OR_RETURN(
      auto rp,
      partition_->AllocateColor("replicated-params", devices08, context04_ptr));
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

  MpmdShardingPropagation propagation{
      partition_.get(), MpmdShardingPropagation::PropagationMode::ForwardFull};
  TF_ASSIGN_OR_RETURN(bool changed, propagation.Run(module.get()));
}

constexpr absl::string_view kDoNotPropagateToParametersHlo = R"(
HloModule jit_raw_generate_synthetic_data, entry_computation_layout={(s32[8,129]{1,0}, s32[8,129]{1,0}, s32[8,128]{1,0})->(s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0})}, allow_spmd_sharding_propagation_to_parameters={true,true,true}, allow_spmd_sharding_propagation_to_output={false,false,false,false,false,false}, num_partitions=8

%sharding (Arg_0.0: s32[8,129], Arg_1.0: s32[8,129], Arg_2.0: s32[8,128]) -> (s32[8,128], s32[8,128], s32[8,128], s32[8,128], s32[8,128], /*index=5*/s32[8,128]) {
  %Arg_0.0 = s32[8,129]{1,0} parameter(0), frontend_attributes={color="sharding"}
  %slice.0 = s32[8,128]{1,0} slice(s32[8,129]{1,0} %Arg_0.0), slice={[0:8], [0:128]}, frontend_attributes={color="sharding"}
  %reshape.0 = s32[8,128]{1,0} reshape(s32[8,128]{1,0} %slice.0), sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %Arg_1.0 = s32[8,129]{1,0} parameter(1), frontend_attributes={color="sharding"}
  %slice.1 = s32[8,128]{1,0} slice(s32[8,129]{1,0} %Arg_1.0), slice={[0:8], [0:128]}, frontend_attributes={color="sharding"}
  %reshape.1 = s32[8,128]{1,0} reshape(s32[8,128]{1,0} %slice.1), sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %Arg_2.0 = s32[8,128]{1,0} parameter(2), frontend_attributes={color="sharding"}
  %reshape.2 = s32[8,128]{1,0} reshape(s32[8,128]{1,0} %Arg_2.0), sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %copy.2 = s32[8,128]{1,0} copy(s32[8,128]{1,0} %reshape.2), sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %slice.2 = s32[8,128]{1,0} slice(s32[8,129]{1,0} %Arg_0.0), slice={[0:8], [1:129]}, frontend_attributes={color="sharding"}
  %reshape.3 = s32[8,128]{1,0} reshape(s32[8,128]{1,0} %slice.2), sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %slice.3 = s32[8,128]{1,0} slice(s32[8,129]{1,0} %Arg_1.0), slice={[0:8], [1:129]}, frontend_attributes={color="sharding"}
  %reshape.4 = s32[8,128]{1,0} reshape(s32[8,128]{1,0} %slice.3), sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %reshape.5 = s32[8,128]{1,0} reshape(s32[8,128]{1,0} %Arg_2.0), sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %copy.3 = s32[8,128]{1,0} copy(s32[8,128]{1,0} %reshape.5), sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  ROOT %tuple = (s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0}) tuple(s32[8,128]{1,0} %reshape.0, s32[8,128]{1,0} %reshape.1, s32[8,128]{1,0} %copy.2, s32[8,128]{1,0} %reshape.3, s32[8,128]{1,0} %reshape.4, /*index=5*/s32[8,128]{1,0} %copy.3)
}

ENTRY %main.15 (Arg_0.1: s32[8,129], Arg_1.2: s32[8,129], Arg_2.3: s32[8,128]) -> (s32[8,128], s32[8,128], s32[8,128], s32[8,128], s32[8,128], /*index=5*/s32[8,128]) {
  %Arg_0.1 = s32[8,129]{1,0} parameter(0), frontend_attributes={color="sharding"}, metadata={op_name="data[0]"}
  %Arg_1.2 = s32[8,129]{1,0} parameter(1), frontend_attributes={color="sharding"}, metadata={op_name="data[1]"}
  %Arg_2.3 = s32[8,128]{1,0} parameter(2), frontend_attributes={color="sharding"}, metadata={op_name="data[2]"}
  %call = (s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0}) call(s32[8,129]{1,0} %Arg_0.1, s32[8,129]{1,0} %Arg_1.2, s32[8,128]{1,0} %Arg_2.3), to_apply=%sharding, frontend_attributes={color="sharding"}
  %get-tuple-element = s32[8,128]{1,0} get-tuple-element((s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0}) %call), index=0, sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %get-tuple-element.1 = s32[8,128]{1,0} get-tuple-element((s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0}) %call), index=1, sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %get-tuple-element.2 = s32[8,128]{1,0} get-tuple-element((s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0}) %call), index=2, sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %get-tuple-element.3 = s32[8,128]{1,0} get-tuple-element((s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0}) %call), index=3, sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %get-tuple-element.4 = s32[8,128]{1,0} get-tuple-element((s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0}) %call), index=4, sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  %get-tuple-element.5 = s32[8,128]{1,0} get-tuple-element((s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0}) %call), index=5, sharding={devices=[8,1]<=[8]}, frontend_attributes={color="sharding"}
  ROOT %tuple.14 = (s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, s32[8,128]{1,0}, /*index=5*/s32[8,128]{1,0}) tuple(s32[8,128]{1,0} %get-tuple-element, s32[8,128]{1,0} %get-tuple-element.1, s32[8,128]{1,0} %get-tuple-element.2, s32[8,128]{1,0} %get-tuple-element.3, s32[8,128]{1,0} %get-tuple-element.4, /*index=5*/s32[8,128]{1,0} %get-tuple-element.5), sharding={{devices=[8,1]<=[8]}, {devices=[8,1]<=[8]}, {devices=[8,1]<=[8]}, {devices=[8,1]<=[8]}, {devices=[8,1]<=[8]}, /*index=5*/{devices=[8,1]<=[8]}}, frontend_attributes={color="sharding"}
}
)";

}  // namespace
}  // namespace xla
