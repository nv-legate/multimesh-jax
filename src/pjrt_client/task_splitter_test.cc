/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/task_splitter.h"

#include "gmock/gmock.h"
#include "mpmd_test_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_coloring.h"
#include "xla/pjrt/multimesh/mpmd_computation_fusion.h"
#include "xla/pjrt/multimesh/mpmd_computation_inliner.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class TaskSplitterTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

using ::testing::Ge;

static constexpr absl::string_view kBasicHlo = R"(
task_f {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  param.1 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  param.2 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_f"}
  param.3 = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_f"}
  add.0 = f32[4,4]{1,0} add(param.0, param.1), frontend_attributes={color="task_f"}
  add.1 = f32[4,4]{1,0} add(add.0, param.1), frontend_attributes={color="task_f"}
  add.2 = f32[4,4]{1,0} add(add.0, param.0), frontend_attributes={color="task_g"}
  add.3 = f32[4,4]{1,0} add(param.2, param.3), frontend_attributes={color="task_g"}
  add.4 = f32[4,4]{1,0} add(add.3, param.2), frontend_attributes={color="task_g"}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.1, add.4)
}

ENTRY main {
  arg_tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  Arg.0 = f32[4,4]{1,0} get-tuple-element(arg_tuple), index=0
  Arg.1 = f32[4,4]{1,0} get-tuple-element(arg_tuple), index=1
  Arg.2 = f32[4,4]{1,0} get-tuple-element(arg_tuple), index=2
  Arg.3 = f32[4,4]{1,0} get-tuple-element(arg_tuple), index=3
  call = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg.0, Arg.1, Arg.2, Arg.3), to_apply=task_f, frontend_attributes={color="task_f"}, metadata={op_name="transpose(jvp"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call), index=0, frontend_attributes={color="task_f"}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call), index=1, frontend_attributes={color="task_f"}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.0, get-tuple-element.1)
}
)";

TEST_F(TaskSplitterTest, BasicSplit) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetHloModuleFromText(kBasicHlo, /*num_devices=*/2));

  TF_ASSERT_OK_AND_ASSIGN(auto color_f,
                          partition_->AllocateColor(
                              "task_f", {{.start = 0, .num_devices = 1}}, {}));

  TF_ASSERT_OK_AND_ASSIGN(auto color_g,
                          partition_->AllocateColor(
                              "task_g", {{.start = 1, .num_devices = 1}}, {}));

  auto* call = module->entry_computation()->GetInstructionWithName("call");
  ASSERT_NE(call, nullptr);

  std::vector<HloInstruction*> f;
  std::vector<HloInstruction*> g;
  for (auto* instruction : call->to_apply()->MakeInstructionPostOrder()) {
    std::string color = *Color(instruction);
    if (color == "task_f") {
      f.push_back(instruction);
    } else if (color == "task_g") {
      g.push_back(instruction);
    }
  }

  TF_ASSERT_OK(SplitTask(call, {f, g}, {"f", "g"}, partition_.get()));

  EXPECT_THAT(EntryComputationCallInstructionLists(module.get()),
              ElementsAre(UnorderedElementsAre(op::Parameter(), op::Parameter(),
                                               op::Add(), op::Add(),
                                               op::Tuple(op::Add(), op::Add())),
                          UnorderedElementsAre(
                              op::Parameter(), op::Parameter(), op::Parameter(),
                              op::Parameter(), op::Parameter(), op::Add(),
                              op::Add(), op::Add(), op::Tuple(op::Add()))));
}

}  // namespace
}  // namespace xla
