/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_instruction_delay_recolor.h"

#include "gmock/gmock.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdInstructionDelayRecolorTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

static constexpr absl::string_view kDelayInitInstructionsHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[8]{0}, f32[8]{0}, f32[8]{0}, f32[8]{0})->(f32[8]{0}, f32[8]{0})}

init0 {
  Arg_0.2 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  Arg_1.3 = f32[8]{0} parameter(1), frontend_attributes={color="red"}
  Arg_2.4 = f32[8]{0} parameter(2), frontend_attributes={color="red"}
  add.0 = f32[8]{0} add(Arg_0.2, Arg_1.3), frontend_attributes={color="red"}
  exp.0 = f32[8]{0} exponential(Arg_2.4), frontend_attributes={color="red"}
  ROOT tuple.0 = (f32[8]{0}, f32[8]{0}) tuple(add.0, exp.0)
}

init1 {
  Arg_0.3 = f32[8]{0} parameter(0), frontend_attributes={color="blue"}
  Arg_1.4 = f32[8]{0} parameter(1), frontend_attributes={color="blue"}
  Arg_2.5 = f32[8]{0} parameter(2), frontend_attributes={color="blue"}
  add.1 = f32[8]{0} add(Arg_0.3, Arg_1.4), frontend_attributes={color="blue"}
  exp.1 = f32[8]{0} exponential(Arg_2.5), frontend_attributes={color="blue"}
  ROOT tuple.1 = (f32[8]{0}, f32[8]{0}) tuple(add.1, exp.1)
}

compute0 {
  Arg_0.4 = f32[8]{0} parameter(0), frontend_attributes={color="green"}
  Arg_1.5 = f32[8]{0} parameter(1), frontend_attributes={color="green"}
  Arg_2.6 = f32[8]{0} parameter(2), frontend_attributes={color="green"}
  add.2 = f32[8]{0} add(Arg_0.4, Arg_1.5), frontend_attributes={color="green"}
  exp.2 = f32[8]{0} exponential(Arg_2.6), frontend_attributes={color="green"}
  out.2 = f32[8]{0} add(exp.2, add.2), frontend_attributes={color="green"}
  ROOT tuple.2 = (f32[8]{0}) tuple(out.2)
}

compute1 {
  Arg_0.5 = f32[8]{0} parameter(0), frontend_attributes={color="yellow"}
  Arg_1.6 = f32[8]{0} parameter(1), frontend_attributes={color="yellow"}
  Arg_2.7 = f32[8]{0} parameter(2), frontend_attributes={color="yellow"}
  add.3 = f32[8]{0} add(Arg_0.5, Arg_1.6), frontend_attributes={color="yellow"}
  exp.3 = f32[8]{0} exponential(Arg_2.7), frontend_attributes={color="yellow"}
  out.3 = f32[8]{0} add(exp.3, add.3), frontend_attributes={color="yellow"}
  ROOT tuple.2 = (f32[8]{0}) tuple(out.3)
}

update0 {
  Arg_0.6 = f32[8]{0} parameter(0), frontend_attributes={color="brown"}
  Arg_1.7 = f32[8]{0} parameter(1), frontend_attributes={color="brown"}
  add.4 = f32[8]{0} add(Arg_0.6, Arg_1.7), frontend_attributes={color="brown"}
  ROOT tuple.4 = (f32[8]{0}) tuple(add.4)
}

update1 {
  Arg_0.7 = f32[8]{0} parameter(0), frontend_attributes={color="orange"}
  Arg_1.8 = f32[8]{0} parameter(1), frontend_attributes={color="orange"}
  add.5 = f32[8]{0} add(Arg_0.7, Arg_1.8), frontend_attributes={color="orange"}
  ROOT tuple.5 = (f32[8]{0}) tuple(add.5)
}


ENTRY main.117 {
  Arg_0.1 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  Arg_1.2 = f32[8]{0} parameter(1), frontend_attributes={color="red"}
  Arg_2.3 = f32[8]{0} parameter(2), frontend_attributes={color="red"}
  call.0 = (f32[8]{0}, f32[8]{0}) call(Arg_0.1, Arg_1.2, Arg_2.3), to_apply=init0, frontend_attributes={color="red"}
  get-tuple-element.0 = f32[8]{0} get-tuple-element(call.0), index=0, frontend_attributes={color="red"}
  get-tuple-element.1 = f32[8]{0} get-tuple-element(call.0), index=1, frontend_attributes={color="red"}
  Arg_3.4 = f32[8]{0} parameter(3), frontend_attributes={color="blue"}
  Arg_4.5 = f32[8]{0} parameter(4), frontend_attributes={color="blue"}
  Arg_5.6 = f32[8]{0} parameter(5), frontend_attributes={color="blue"}
  call.1 = (f32[8]{0}, f32[8]{0}) call(Arg_3.4, Arg_4.5, Arg_5.6), to_apply=init1, frontend_attributes={color="blue"}
  get-tuple-element.2 = f32[8]{0} get-tuple-element(call.1), index=0, frontend_attributes={color="blue"}
  get-tuple-element.3 = f32[8]{0} get-tuple-element(call.1), index=1, frontend_attributes={color="blue"}
  call.2 = (f32[8]{0}) call(Arg_0.1, Arg_1.2, get-tuple-element.0), to_apply=compute0, frontend_attributes={color="green"}
  call.3 = (f32[8]{0}) call(Arg_2.3, Arg_3.4, get-tuple-element.2), to_apply=compute1, frontend_attributes={color="yellow"}
  get-tuple-element.4 = f32[8]{0} get-tuple-element(call.2), index=0, frontend_attributes={color="green"}
  get-tuple-element.5 = f32[8]{0} get-tuple-element(call.3), index=0, frontend_attributes={color="yellow"}
  call.4 = (f32[8]{0}) call(get-tuple-element.4, get-tuple-element.1), to_apply=update0, frontend_attributes={color="brown"}
  call.5 = (f32[8]{0}) call(get-tuple-element.5, get-tuple-element.3), to_apply=update1, frontend_attributes={color="orange"}
  gte.2 = f32[8]{0} get-tuple-element(call.4), index=0, frontend_attributes={color="brown"}
  gte.3 = f32[8]{0} get-tuple-element(call.5), index=0, frontend_attributes={color="orange"}
  ROOT tuple = (f32[8]{0}, f32[8]{0}) tuple(gte.2, gte.3)
} // main.117
)";

TEST_F(MpmdInstructionDelayRecolorTest, DelayInitInstructions) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kDelayInitInstructionsHlo, /*num_devices=*/2));

  TF_ASSIGN_OR_RETURN(auto green,
                      partition_->AllocateColor(
                          "green", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto red,
                      partition_->AllocateColor(
                          "red", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto orange,
                      partition_->AllocateColor(
                          "orange", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto yellow,
                      partition_->AllocateColor(
                          "yellow", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto brown,
                      partition_->AllocateColor(
                          "brown", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto blue,
                      partition_->AllocateColor(
                          "blue", {{.start = 0, .num_devices = 2}}, nullptr));

  MpmdInstructionDelayRecolor recolor{partition_.get()};

  TF_ASSERT_OK_AND_ASSIGN(bool changed, recolor.Run(module.get()));

  // the orange and brown computations should now have the cosine operation
  EXPECT_THAT(
      module->computations(),
      AllOf(
          Not(Contains(Property(&HloComputation::instructions,
                                Contains(AllOf(op::Exp(), m::Color("blue")))))),
          Contains(Property(&HloComputation::instructions,
                            Contains(AllOf(op::Exp(), m::Color("orange")))))));
}

static constexpr absl::string_view kSharedConstantHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[8]{0}, f32[8]{0}, f32[8]{0}, f32[8]{0})->(f32[8]{0}, f32[8]{0})}

init0 {
  Arg_0.2 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  Arg_1.3 = f32[8]{0} parameter(1), frontend_attributes={color="red"}
  Arg_2.4 = f32[8]{0} parameter(2), frontend_attributes={color="red"}
  constant.0 = f32[] constant(0.0)
  broadcast.0 = f32[8]{0} broadcast(constant.0), dimensions={}
  add.0 = f32[8]{0} add(Arg_0.2, Arg_1.3), frontend_attributes={color="red"}
  exp.0 = f32[8]{0} exponential(Arg_2.4), frontend_attributes={color="red"}
  ROOT tuple.0 = (f32[8]{0}, f32[8]{0}) tuple(add.0, exp.0)
}

init1 {
  Arg_0.3 = f32[8]{0} parameter(0), frontend_attributes={color="blue"}
  Arg_1.4 = f32[8]{0} parameter(1), frontend_attributes={color="blue"}
  Arg_2.5 = f32[8]{0} parameter(2), frontend_attributes={color="blue"}
  constant.1 = f32[] constant(0.0)
  broadcast.1 = f32[8]{0} broadcast(constant.1), dimensions={}
  add.1 = f32[8]{0} add(Arg_0.3, Arg_1.4), frontend_attributes={color="blue"}
  add.11 = f32[8]{0} add(add.1, broadcast.1)
  exp.1 = f32[8]{0} exponential(Arg_2.5), frontend_attributes={color="blue"}
  ROOT tuple.1 = (f32[8]{0}, f32[8]{0}) tuple(add.1, exp.1)
}

compute0 {
  Arg_0.4 = f32[8]{0} parameter(0), frontend_attributes={color="green"}
  Arg_1.5 = f32[8]{0} parameter(1), frontend_attributes={color="green"}
  Arg_2.6 = f32[8]{0} parameter(2), frontend_attributes={color="green"}
  add.2 = f32[8]{0} add(Arg_0.4, Arg_1.5), frontend_attributes={color="green"}
  exp.2 = f32[8]{0} exponential(Arg_2.6), frontend_attributes={color="green"}
  out.2 = f32[8]{0} add(exp.2, add.2), frontend_attributes={color="green"}
  ROOT tuple.2 = (f32[8]{0}) tuple(out.2)
}

compute1 {
  Arg_0.5 = f32[8]{0} parameter(0), frontend_attributes={color="yellow"}
  Arg_1.6 = f32[8]{0} parameter(1), frontend_attributes={color="yellow"}
  Arg_2.7 = f32[8]{0} parameter(2), frontend_attributes={color="yellow"}
  add.3 = f32[8]{0} add(Arg_0.5, Arg_1.6), frontend_attributes={color="yellow"}
  exp.3 = f32[8]{0} exponential(Arg_2.7), frontend_attributes={color="yellow"}
  out.3 = f32[8]{0} add(exp.3, add.3), frontend_attributes={color="yellow"}
  ROOT tuple.2 = (f32[8]{0}) tuple(out.3)
}

update0 {
  Arg_0.6 = f32[8]{0} parameter(0), frontend_attributes={color="brown"}
  Arg_1.7 = f32[8]{0} parameter(1), frontend_attributes={color="brown"}
  add.4 = f32[8]{0} add(Arg_0.6, Arg_1.7), frontend_attributes={color="brown"}
  ROOT tuple.4 = (f32[8]{0}) tuple(add.4)
}

update1 {
  Arg_0.7 = f32[8]{0} parameter(0), frontend_attributes={color="orange"}
  Arg_1.8 = f32[8]{0} parameter(1), frontend_attributes={color="orange"}
  add.5 = f32[8]{0} add(Arg_0.7, Arg_1.8), frontend_attributes={color="orange"}
  ROOT tuple.5 = (f32[8]{0}) tuple(add.5)
}


ENTRY main.117 {
  Arg_0.1 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  Arg_1.2 = f32[8]{0} parameter(1), frontend_attributes={color="red"}
  Arg_2.3 = f32[8]{0} parameter(2), frontend_attributes={color="red"}
  call.0 = (f32[8]{0}, f32[8]{0}) call(Arg_0.1, Arg_1.2, Arg_2.3), to_apply=init0, frontend_attributes={color="red"}
  get-tuple-element.0 = f32[8]{0} get-tuple-element(call.0), index=0, frontend_attributes={color="red"}
  get-tuple-element.1 = f32[8]{0} get-tuple-element(call.0), index=1, frontend_attributes={color="red"}
  Arg_3.4 = f32[8]{0} parameter(3), frontend_attributes={color="blue"}
  Arg_4.5 = f32[8]{0} parameter(4), frontend_attributes={color="blue"}
  Arg_5.6 = f32[8]{0} parameter(5), frontend_attributes={color="blue"}
  call.1 = (f32[8]{0}, f32[8]{0}) call(Arg_3.4, Arg_4.5, Arg_5.6), to_apply=init1, frontend_attributes={color="blue"}
  get-tuple-element.2 = f32[8]{0} get-tuple-element(call.1), index=0, frontend_attributes={color="blue"}
  get-tuple-element.3 = f32[8]{0} get-tuple-element(call.1), index=1, frontend_attributes={color="blue"}
  call.2 = (f32[8]{0}) call(Arg_0.1, Arg_1.2, get-tuple-element.0), to_apply=compute0, frontend_attributes={color="green"}
  call.3 = (f32[8]{0}) call(Arg_2.3, Arg_3.4, get-tuple-element.2), to_apply=compute1, frontend_attributes={color="yellow"}
  get-tuple-element.4 = f32[8]{0} get-tuple-element(call.2), index=0, frontend_attributes={color="green"}
  get-tuple-element.5 = f32[8]{0} get-tuple-element(call.3), index=0, frontend_attributes={color="yellow"}
  call.4 = (f32[8]{0}) call(get-tuple-element.4, get-tuple-element.1), to_apply=update0, frontend_attributes={color="brown"}
  call.5 = (f32[8]{0}) call(get-tuple-element.5, get-tuple-element.3), to_apply=update1, frontend_attributes={color="orange"}
  gte.2 = f32[8]{0} get-tuple-element(call.4), index=0, frontend_attributes={color="brown"}
  gte.3 = f32[8]{0} get-tuple-element(call.5), index=0, frontend_attributes={color="orange"}
  ROOT tuple = (f32[8]{0}, f32[8]{0}) tuple(gte.2, gte.3)
} // main.117
)";

TEST_F(MpmdInstructionDelayRecolorTest, DelaySharedBroadcastInstructions) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kSharedConstantHlo, /*num_devices=*/2));

  TF_ASSIGN_OR_RETURN(auto green,
                      partition_->AllocateColor(
                          "green", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto red,
                      partition_->AllocateColor(
                          "red", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto orange,
                      partition_->AllocateColor(
                          "orange", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto yellow,
                      partition_->AllocateColor(
                          "yellow", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto brown,
                      partition_->AllocateColor(
                          "brown", {{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(auto blue,
                      partition_->AllocateColor(
                          "blue", {{.start = 0, .num_devices = 2}}, nullptr));

  MpmdInstructionDelayRecolor recolor{partition_.get()};

  TF_ASSERT_OK_AND_ASSIGN(bool changed, recolor.Run(module.get()));

  // the orange and brown computations should now have the cosine operation
  EXPECT_THAT(
      module->computations(),
      AllOf(
          Not(Contains(Property(&HloComputation::instructions,
                                Contains(AllOf(op::Exp(), m::Color("blue")))))),
          Contains(Property(
              &HloComputation::instructions,
              Contains(AllOf(op::Exp(), m::Color("orange"))).Times(1)))));
}

}  // namespace
}  // namespace xla
