/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_loop_unroll.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::Field;
using ::testing::Not;

class MpmdLoopUnrollTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

static constexpr absl::string_view kBasicLoopHlo = R"(
task_f {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  param.1 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  add.0 = f32[4,4]{1,0} add(param.0, param.1), frontend_attributes={color="task_f"}
  add.1 = f32[4,4]{1,0} add(add.0, param.1), frontend_attributes={color="task_f"}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.0, add.1)
}

task_g {
  param.2 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}
  param.3 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}
  param.4 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_g"}
  param.5  = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_g"}
  add.4 = f32[4,4]{1,0} add(param.2, param.3), frontend_attributes={color="task_g"}
  add.5 = f32[4,4]{1,0} add(param.4, param.5), frontend_attributes={color="task_g"}
  ROOT tuple.4 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.4, add.5)
}

task_init {
  zero = f32[] constant(0.0)
  broadcast.0 = f32[4,4]{1,0} broadcast(zero), dimensions={}
  broadcast.1 = f32[4,4]{1,0} broadcast(zero), dimensions={}
  ROOT tuple.1 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(broadcast.0, broadcast.1)
}

loop {
  arg_tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  Arg.0 = f32[4,4]{1,0} get-tuple-element(arg_tuple), index=0
  Arg.1 = f32[4,4]{1,0} get-tuple-element(arg_tuple), index=1
  Arg.2 = f32[4,4]{1,0} get-tuple-element(arg_tuple), index=2
  Arg.3 = f32[4,4]{1,0} get-tuple-element(arg_tuple), index=3
  call.2 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg.0, Arg.1), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call.2), index=1, frontend_attributes={color="task_f"}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(get-tuple-element.0, get-tuple-element.1, Arg.2, Arg.3), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.3), index=1, frontend_attributes={color="task_g"}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(Arg.0, Arg.1, get-tuple-element.2, get-tuple-element.3)
}

condition {
  arg_tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  ROOT constant = pred[] constant(true)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0)
  Arg_1.2 = f32[4,4]{1,0} parameter(1)
  call.0 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(), to_apply=task_init, frontend_attributes={color="task_f"}
  get-tuple-element.4 = f32[4,4]{1,0} get-tuple-element(call.0), index=0
  get-tuple-element.5 = f32[4,4]{1,0} get-tuple-element(call.0), index=1
  tuple.0 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(Arg_0.1, Arg_1.2, get-tuple-element.4, get-tuple-element.5)
  while.0 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) while(tuple.0), condition=condition, body=loop, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  get-tuple-element.6 = f32[4,4]{1,0} get-tuple-element(while.0), index=2, frontend_attributes={color="task_g"}
  get-tuple-element.7 = f32[4,4]{1,0} get-tuple-element(while.0), index=3, frontend_attributes={color="task_g"}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.6, get-tuple-element.7)
}
)";

TEST_F(MpmdLoopUnrollTest, BasicLoop) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kBasicLoopHlo, /*num_devices=*/1));

  // body, condition, f, g, init, entry
  EXPECT_EQ(module->computation_count(), 6);

  MpmdLoopUnroll unroller{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, unroller.Run(module.get()));

  // f, g, init, entry
  EXPECT_EQ(module->computation_count(), 4);

  EXPECT_THAT(
      module->entry_computation()->instructions(),
      AllOf(Contains(op::Call()).Times(5),     // init, 2xf, 2xg
            Contains(op::Tuple()).Times(1)));  // only root tuple is left

  // each of the loop-carried variables should have pre-loop init
  // and each of the two post-task values
  EXPECT_THAT(
      module->entry_computation()->instructions(),
      AllOf(
          Contains(m::MetadataSchedulingNames("get-tuple-element.4")).Times(3),
          Contains(m::MetadataSchedulingNames("get-tuple-element.5"))
              .Times(3)));

  HloPrintOptions options = HloPrintOptions::Default();
  options.set_print_control_dependencies(true);
}

}  // namespace
}  // namespace xla
