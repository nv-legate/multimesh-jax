/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_unused_param_output_remover.h"

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdUnusedParamOutputRemoverTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

using ::testing::Ge;
using ::testing::Lt;

constexpr absl::string_view kRootParameterHlo = R"(
HloModule jit_c, num_partitions=2

task_f {
  get-tuple-element.4 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  get-tuple-element.5 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  add.0 = f32[4,4]{1,0} add(get-tuple-element.4, get-tuple-element.5), frontend_attributes={color="task_f"}
  add.1 = f32[4,4]{1,0} add(add.0, get-tuple-element.5), frontend_attributes={color="task_f"}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.0, add.1)
}

task_g {
  get-tuple-element.7 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}
  get-tuple-element.8 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}
  get-tuple-element.9 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_g"}
  add.4 = f32[4,4]{1,0} add(get-tuple-element.8, get-tuple-element.9), frontend_attributes={color="task_g"}
  add.5 = f32[4,4]{1,0} add(add.4, get-tuple-element.8), frontend_attributes={color="task_g"}
  ROOT tuple.4 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.7, add.5)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0)
  Arg_1.2 = f32[4,4]{1,0} parameter(1)
  Arg_2.3 = f32[4,4]{1,0} parameter(2)
  call.2 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call.2), index=1, frontend_attributes={color="task_f"}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(get-tuple-element.0, get-tuple-element.1, Arg_2.3), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.3), index=1, frontend_attributes={color="task_g"}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.2, get-tuple-element.3)
}
)";

TEST_F(MpmdUnusedParamOutputRemoverTest, RootParameter) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kRootParameterHlo, /*num_devices=*/2));

  Shape unit_shape =
      module->entry_computation()->parameter_instruction(0)->shape();
  Shape tuple1_shape{std::vector<Shape>{unit_shape}};
  Shape tuple2_shape{{unit_shape, unit_shape}};

  EXPECT_THAT(
      m::EntryComputationCalls(module.get()),
      ElementsAre(AllOf(op::Call(op::Parameter(), op::Parameter()),
                        op::Shape(tuple2_shape), m::Color("task_f")),
                  AllOf(op::Call(op::GetTupleElement(), op::GetTupleElement(),
                                 op::Parameter()),
                        op::Shape(tuple2_shape), m::Color("task_g"))));

  MpmdUnusedParamOutputRemover remover{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, remover.Run(module.get()));

  EXPECT_THAT(
      m::EntryComputationCalls(module.get()),
      ElementsAre(AllOf(op::Call(op::Parameter(), op::Parameter()),
                        op::Shape(tuple2_shape), m::Color("task_f")),
                  AllOf(op::Call(op::GetTupleElement(), op::Parameter()),
                        op::Shape(tuple1_shape), m::Color("task_g"))));
}

constexpr absl::string_view kURootParameterUsedInComputationHlo = R"(
HloModule jit_c, num_partitions=2

task_f {
  get-tuple-element.4 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  get-tuple-element.5 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  add.0 = f32[4,4]{1,0} add(get-tuple-element.4, get-tuple-element.5), frontend_attributes={color="task_f"}
  add.1 = f32[4,4]{1,0} add(add.0, get-tuple-element.5), frontend_attributes={color="task_f"}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.0, add.1)
}

task_g {
  get-tuple-element.7 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}
  get-tuple-element.8 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}
  get-tuple-element.9 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_g"}
  add.4 = f32[4,4]{1,0} add(get-tuple-element.8, get-tuple-element.7), frontend_attributes={color="task_g"}
  add.5 = f32[4,4]{1,0} add(add.4, get-tuple-element.9), frontend_attributes={color="task_g"}
  ROOT tuple.4 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.7, add.5)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0)
  Arg_1.2 = f32[4,4]{1,0} parameter(1)
  Arg_2.3 = f32[4,4]{1,0} parameter(2)
  call.2 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call.2), index=1, frontend_attributes={color="task_f"}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(get-tuple-element.0, get-tuple-element.1, Arg_2.3), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.3), index=1, frontend_attributes={color="task_g"}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.2, get-tuple-element.3)
}
)";

TEST_F(MpmdUnusedParamOutputRemoverTest, RootParameterUsedInComputation) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kURootParameterUsedInComputationHlo,
                                        /*num_devices=*/2));

  Shape unit_shape =
      module->entry_computation()->parameter_instruction(0)->shape();
  Shape tuple1_shape{std::vector<Shape>{unit_shape}};
  Shape tuple2_shape{{unit_shape, unit_shape}};

  EXPECT_THAT(
      m::EntryComputationCalls(module.get()),
      ElementsAre(AllOf(op::Call(op::Parameter(), op::Parameter()),
                        op::Shape(tuple2_shape), m::Color("task_f")),
                  AllOf(op::Call(op::GetTupleElement(), op::GetTupleElement(),
                                 op::Parameter()),
                        op::Shape(tuple2_shape), m::Color("task_g"))));

  MpmdUnusedParamOutputRemover remover{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, remover.Run(module.get()));

  // the second call should not produce 2 outputs, but
  // should still have three inputs
  EXPECT_THAT(
      m::EntryComputationCalls(module.get()),
      ElementsAre(AllOf(op::Call(op::Parameter(), op::Parameter()),
                        op::Shape(tuple2_shape), m::Color("task_f")),
                  AllOf(op::Call(op::GetTupleElement(), op::GetTupleElement(),
                                 op::Parameter()),
                        op::Shape(tuple1_shape), m::Color("task_g"))));
}

}  // namespace
}  // namespace xla
