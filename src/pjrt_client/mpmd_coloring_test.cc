/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_coloring.h"

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_microbatch_loop_canonicalizer.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdColoringTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

using ::testing::Gt;

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

TEST_F(MpmdColoringTest, SimpleImplicitTask) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kSimpleImplicitTaskHlo, /*num_devices=*/4));

  RegisterMatcherTestTask("(task_f)", {0, 1}, {2}, {"x"}, {{"x", "batch"}});
  RegisterMatcherTestTask("(task_g)", {2, 3}, {2}, {"x"}, {{"x", "batch"}});

  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  // f -> color 0
  // g -> color 1

  // multiply.4, multiply.7, 2 Aargs, a convert and a broadcast
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(m::Color("task_f")).Times(6));

  // cosine.8, add.9, reduce.14
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(m::Color("task_g")).Times(3));

  // all adds should have a color
  EXPECT_THAT(module->entry_computation()->instructions(),
              Each(AnyOf(AllOf(op::Add(), m::HasColor()), Not(op::Add()))));
}

static constexpr absl::string_view kTaskWithBarrierHlo = R"(
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

TEST_F(MpmdColoringTest, TaskWithBarrier) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetHloModuleFromText(kTaskWithBarrierHlo,
                                                            /*num_devices=*/4));

  RegisterMatcherTestTask("(task_f)", {0, 1}, {2}, {"x"}, {{"x", "batch"}});
  RegisterMatcherTestTask("(task_g)", {2, 3}, {2}, {"x"}, {{"x", "batch"}});

  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  // f -> color 0
  // g -> color 1

  // one of the barriers should have a color, the other should not
  EXPECT_THAT(
      module->entry_computation()->instructions(),
      Contains(AllOf(op::OptimizationBarrier(), m::HasColor())).Times(1));
  EXPECT_THAT(
      module->entry_computation()->instructions(),
      Contains(AllOf(op::OptimizationBarrier(), Not(m::HasColor()))).Times(1));

  // one of the tuples should have a color, the other should not
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(AllOf(op::Tuple(), m::HasColor())).Times(1));
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(AllOf(op::Tuple(), Not(m::HasColor()))).Times(1));

  // all of the adds and cosines should have a color
  EXPECT_THAT(module->entry_computation()->instructions(),
              Not(Contains(AllOf(op::Add(), Not(m::HasColor())))));
}

static constexpr absl::string_view kNestedWhileHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4]{0}, f32[4]{0})->f32[]}

region_0.83 {
  arg_tuple.84 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  get-tuple-element.10 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=0
  get-tuple-element.11 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=1
  get-tuple-element.12 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=2
  get-tuple-element.13 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=3
  add.10 = f32[4,4]{1,0} add(get-tuple-element.10, get-tuple-element.12)
  add.11 = f32[4,4]{1,0} add(get-tuple-element.11, get-tuple-element.13), frontend_attributes={color="task_f"}
  add.12 = f32[4,4]{1,0} add(add.10, get-tuple-element.12), frontend_attributes={color="task_g"}
  add.13 = f32[4,4]{1,0} add(add.11, get-tuple-element.13)
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.10, get-tuple-element.11, add.12, add.13)
} // region_0.83

region_2.98 {
  arg_tuple.99 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  constant.106 = s32[] constant(2)
  constant.107 = s32[] constant(2)
  ROOT compare.108 = pred[] compare(constant.106, constant.107), direction=LT
} // region_2.98

ENTRY main.117 {
  constant.6 = f32[] constant(0)
  broadcast.6 = f32[4,4] broadcast(constant.6), dimensions={}
  broadcast.7 = f32[4,4] broadcast(constant.6), dimensions={}
  Arg_0.1 = f32[4,4]{1,0} parameter(0)
  Arg_1.2 = f32[4,4]{1,0} parameter(1)
  Arg_2.3 = f32[4,4]{1,0} parameter(2)
  Arg_3.4 = f32[4,4]{1,0} parameter(3)
  tuple.10 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(Arg_0.1, Arg_1.2, broadcast.6, broadcast.7)
  while.109 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) while(tuple.10), condition=region_2.98, body=region_0.83
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(while.109), index=2
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(while.109), index=3
  add.2 = f32[4,4]{1,0} add(get-tuple-element.2, Arg_2.3)
  add.3 = f32[4,4]{1,0} add(get-tuple-element.3, Arg_3.4)
  ROOT tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.2, add.3)
} // main.117
)";

TEST_F(MpmdColoringTest, NestedWhileColoring) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetHloModuleFromText(kNestedWhileHlo,
                                                            /*num_devices=*/4));
  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  // all add operations should have been assigned a color
  EXPECT_THAT(m::FlatInstructions(module.get()),
              Each(Not(AllOf(op::Add(), Not(m::HasColor())))));
}

TEST_F(MpmdColoringTest, TransformerColoring) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetHloModuleFromPath("opt_8layers.txt",
                                                            /*num_devices=*/8));

  auto device_factory = [](const std::string& name) {
    int layer_num;
    bool parsed = absl::SimpleAtoi(name.substr(9), &layer_num);
    if (!parsed) {
      throw std::runtime_error(
          absl::StrCat("failed to parse layer number from", name));
    }
    int64_t offset = (layer_num / 4) * 4;
    std::vector<int64_t> devices(4);
    std::iota(devices.begin(), devices.end(), offset);
    return devices;
  };
  RegisterMatcherTestTaskWithFactory("(x_layers_\\d+)", device_factory, {2, 2},
                                     {"x", "y"},
                                     {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(emb).*", {0, 1, 2, 3, 4, 5, 6, 7}, {2, 4},
                          {"x", "y"}, {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(emb).*", {0, 1, 2, 3, 4, 5, 6, 7}, {2, 4},
                          {"x", "y"}, {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(final_ln).*", {0, 1, 2, 3, 4, 5, 6, 7}, {2, 4},
                          {"x", "y"}, {{"replica", "x"}, {"mdl", "y"}});
  RegisterMatcherTestTask("(compute_loss).*", {0, 1, 2, 3, 4, 5, 6, 7}, {2, 4},
                          {"x", "y"}, {{"replica", "x"}, {"mdl", "y"}});

  MpmdMicrobatchLoopCanonicalizer inliner{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inliner.Run(module.get()));

  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(changed, coloring.Run(module.get()));

  // all add and dot operations and microbatch init (constant copies)
  // should have been assigned a color
  // we should have many dot operations with the different layer colors
  // the corresponding copies creating the microbatch init
  // should also have been assigned the right colors
  EXPECT_THAT(
      m::FlatInstructionsWithoutReducesAndPredicates(module.get()),
      AllOf(Each(Not(AllOf(m::ShapeLargerThan(10000),
                           AnyOf(op::Add(), op::Dot(), op::Copy()),
                           Not(m::HasColor())))),
            Contains(AllOf(op::Dot(), m::Color("emb"))).Times(Gt(0)),
            Contains(AllOf(op::Dot(), m::Color("compute_loss"))).Times(Gt(0)),
            Contains(m::Color("final_ln")).Times(Gt(5)),
            Contains(AllOf(op::Copy(), m::ShapeLargerThan(5000),
                           m::Color("x_layers_0")))
                .Times(Gt(5)),
            Contains(AllOf(op::Dot(), m::Color("x_layers_0"))).Times(Gt(5)),
            Contains(AllOf(op::Copy(), m::ShapeLargerThan(5000),
                           m::Color("x_layers_6")))
                .Times(Gt(5)),
            Contains(AllOf(op::Dot(), m::Color("x_layers_6"))).Times(Gt(5))));
}

static constexpr absl::string_view kBackwardsColoringDoesNotOverwrite = R"(
ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  multiply.4 = f32[8]{0} multiply(Arg_0.1, Arg_0.1), frontend_attributes={color="task_f"}
  Arg_1.2 = s32[] parameter(1), sharding={replicated}
  convert.5 = f32[] convert(Arg_1.2)
  broadcast.6 = f32[8]{0} broadcast(convert.5), dimensions={}
  add.7 = f32[8]{0} add(multiply.4, broadcast.6)
  cosine.8 = f32[8]{0} cosine(add.7)
  add.9 = f32[8]{0} add(cosine.8, add.7), frontend_attributes={color="task_g"}
  ROOT add.10 = f32[8]{0} add(add.9, cosine.8)
} // main.15
)";

TEST_F(MpmdColoringTest, BackwardsColoringDoesNotOverwrite) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kBackwardsColoringDoesNotOverwrite,
                                        /*num_devices=*/4));
  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  // any backwards coloring should not overwrite the assigned color
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(AllOf(op::Multiply(), m::Color("task_f"))).Times(1));
}

TEST_F(MpmdColoringTest, FavorMostRecentOperands) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kBackwardsColoringDoesNotOverwrite,
                                        /*num_devices=*/4));
  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  // the root should be colored based on the most recent operand
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(AllOf(op::Add(), m::Color("task_g"))).Times(2));
}

constexpr absl::string_view kMpmdTaskHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[8]{0}, s32[])->(f32[], f32[8]{0})}, allow_spmd_sharding_propagation_to_parameters={true,true}, allow_spmd_sharding_propagation_to_output={true,true}

f.impl.12 {
  Arg_0.13 = f32[8]{0} parameter(0), metadata={op_name="jit(c)/jit(main)/legate_task"}
  multiply.15 = f32[8]{0} multiply(Arg_0.13, Arg_0.13), metadata={op_name="jit(c)/jit(main)/jvp(jit(f.impl))/mul"}
  Arg_1.14 = s32[] parameter(1), metadata={op_name="jit(c)/jit(main)/legate_task"}
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
  Arg_0.37 = f32[8]{0} parameter(0), metadata={op_name="jit(c)/jit(main)/legate_task"}
  exp.40 = f32[8]{0} exponential(Arg_0.37), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/cos"}
  Arg_1.38 = f32[8]{0} parameter(1), metadata={op_name="jit(c)/jit(main)/legate_task"}
  add.41 = f32[8]{0} add(exp.40, Arg_1.38), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/add"}
  constant.39 = f32[] constant(0)
  ROOT reduce.42 = f32[] reduce(add.41, constant.39), dimensions={0}, to_apply=region_0.32, metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/reduce_sum[axes=(0,)]"}
} // g.impl.36

f_bwd.impl.56 {
  Arg_1.58 = f32[8]{0} parameter(1)
  Arg_2.59 = f32[] parameter(2)
  broadcast.61 = f32[8]{0} broadcast(Arg_2.59), dimensions={}
  negate.62 = f32[8]{0} negate(broadcast.61)
  Arg_0.57 = f32[8]{0} parameter(0)
  sine.60 = f32[8]{0} sine(Arg_0.57)
  multiply.63 = f32[8]{0} multiply(negate.62, sine.60)
  ROOT tuple.64 = (f32[8]{0}, f32[8]{0}) tuple(multiply.63, broadcast.61)
} // f_bwd.impl.56

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
  Arg_0.1 = f32[8]{0} parameter(0)
  Arg_1.2 = s32[] parameter(1)
  custom-call.19 = f32[8]{0} custom-call(Arg_0.1, Arg_1.2), custom_call_target="LegateTask", called_computations={f.impl.12}, backend_config={"name": "f", "devices": [0], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  custom-call.43 = f32[] custom-call(custom-call.19, Arg_0.1), custom_call_target="LegateTask", called_computations={g.impl.36}, backend_config={"name": "g", "devices": [1], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  constant.3 = f32[] constant(1)
  custom-call.65 = (f32[8]{0}, f32[8]{0}) custom-call(custom-call.19, Arg_0.1, constant.3), custom_call_target="LegateTask", called_computations={f_bwd.impl.56}, backend_config={"name": "g.bwd", "devices": [1], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  get-tuple-element.67 = f32[8]{0} get-tuple-element(custom-call.65), index=1
  get-tuple-element.66 = f32[8]{0} get-tuple-element(custom-call.65), index=0
  custom-call.95 = (f32[8]{0}, pred[]) custom-call(Arg_0.1, Arg_1.2, get-tuple-element.66), custom_call_target="LegateTask", called_computations={f_bwd.impl_0.83}, backend_config={"name": "f.bwd", "devices": [0], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  get-tuple-element.96 = f32[8]{0} get-tuple-element(custom-call.95), index=0
  add.98 = f32[8]{0} add(get-tuple-element.67, get-tuple-element.96)
  ROOT tuple.99 = (f32[], f32[8]{0}) tuple(custom-call.43, add.98)
} // main.100
)";

TEST_F(MpmdColoringTest, MpmdTaskCall) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kMpmdTaskHlo, /*num_devices=*/2));

  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  EXPECT_THAT(m::FlatInstructions(module.get()),
              AllOf(Not(Contains(op::CustomCall("LegateTask"))),
                    Contains(AllOf(op::Exp(), m::Color("g"))).Times(1),
                    Contains(AllOf(op::Negate(), m::Color("g.bwd"))).Times(1)));
}

static constexpr absl::string_view kTaskWithSplitBarrierHlo = R"(
region_0.10 {
  Arg_0.11 = f32[] parameter(0)
  Arg_1.12 = f32[] parameter(1)
  ROOT add.13 = f32[] add(Arg_0.11, Arg_1.12), metadata={op_name="jit(c)/jit(main)/task_g/reduce_sum[axes=(0,)]"}
}

ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}
  multiply.4 = f32[8]{0} multiply(Arg_0.1, Arg_0.1), frontend_attributes={color="task_f"}
  Arg_1.2 = s32[] parameter(1), sharding={replicated}
  convert.5 = f32[] convert(Arg_1.2), frontend_attributes={color="task_f"}
  broadcast.6 = f32[8]{0} broadcast(convert.5), dimensions={}, frontend_attributes={color="task_f"}
  multiply.7 = f32[8]{0} multiply(multiply.4, broadcast.6), frontend_attributes={color="task_f"}
  exp.8 = f32[8]{0} exponential(multiply.7), frontend_attributes={color="task_g"}
  add.9 = f32[8]{0} add(exp.8, multiply.7), frontend_attributes={color="task_g"}
  tuple.1 = (f32[8]{0}, f32[8]{0}, f32[8]{0}) tuple(multiply.7, exp.8, add.9)
  opt-barrier.1 = (f32[8]{0}, f32[8]{0}, f32[8]{0}) opt-barrier(tuple.1)
  gte.multiply.7 = get-tuple-element(opt-barrier.1), index=0
  gte.exp.8 = get-tuple-element(opt-barrier.1), index=1
  gte.add.9 = get-tuple-element(opt-barrier.1), index=2
  add.10 = f32[8]{0} add(gte.exp.8, gte.multiply.7), frontend_attributes={color="task_g"}
  tuple.2 = (f32[8]{0}, f32[8]{0}) tuple(add.10, exp.8)
  opt-barrier.2 = (f32[8]{0}, f32[8]{0}) opt-barrier(tuple.2)
  gte.add.10 = get-tuple-element(opt-barrier.2), index=0
  ROOT add.11 = f32[] add(gte.exp.8, gte.add.10)
} // main.15
)";

TEST_F(MpmdColoringTest, TaskWithSplitBarrierHlo) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kTaskWithSplitBarrierHlo, /*num_devices=*/2));

  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  // all maths ops should have a color
  // only one of the 2 opt-barriers should have been colored
  EXPECT_THAT(
      m::FlatInstructionsWithoutReducesAndPredicates(module.get()),
      AllOf(
          Each(Not(AnyOf(AllOf(op::Add(), Not(m::HasColor())),
                         AllOf(op::Exp(), Not(m::HasColor()))))),
          Contains(op::OptimizationBarrier()).Times(2),
          Contains(AllOf(op::OptimizationBarrier(), m::HasColor())).Times(1)));
}

}  // namespace
}  // namespace xla
