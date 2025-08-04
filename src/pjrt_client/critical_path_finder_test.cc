/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/critical_path_finder.h"

#include "critical_path_finder.h"
#include "gmock/gmock.h"
#include "mpmd_test_base.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_instruction.h"
#include "xla/pjrt/multimesh/mpmd_coloring.h"
#include "xla/pjrt/multimesh/mpmd_computation_fusion.h"
#include "xla/pjrt/multimesh/mpmd_computation_inliner.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"
#include "xla/pjrt/multimesh/mpmd_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class CriticalPathFinderTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;

using ::testing::Ge;

static constexpr absl::string_view kBasicHlo = R"(
task_f {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  add.0 = f32[4,4]{1,0} add(param.0, param.0), frontend_attributes={color="task_f"}
  mul.0 = f32[4,4]{1,0} multiply(add.0, add.0), frontend_attributes={color="task_f"}
  sub.0 = f32[4,4]{1,0} subtract(param.0, mul.0), frontend_attributes={color="task_f"}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.0, mul.0, sub.0)
}

ENTRY main {
  arg_tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  Arg.0 = f32[4,4]{1,0} get-tuple-element(arg_tuple), index=0
  call = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg.0), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call), index=0, frontend_attributes={color="task_f"}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call), index=1, frontend_attributes={color="task_f"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call), index=2, frontend_attributes={color="task_f"}
  call.1 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) call(get-tuple-element.1), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.1), index=0, frontend_attributes={color="task_f"}
  get-tuple-element.4 = f32[4,4]{1,0} get-tuple-element(call.1), index=1, frontend_attributes={color="task_f"}
  get-tuple-element.5 = f32[4,4]{1,0} get-tuple-element(call.1), index=2, frontend_attributes={color="task_f"}
  call.2 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) call(get-tuple-element.5), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.6 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}
  get-tuple-element.7 = f32[4,4]{1,0} get-tuple-element(call.2), index=1, frontend_attributes={color="task_f"}
  get-tuple-element.8 = f32[4,4]{1,0} get-tuple-element(call.2), index=2, frontend_attributes={color="task_f"}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.0, get-tuple-element.2, get-tuple-element.3, get-tuple-element.4, get-tuple-element.6, get-tuple-element.7, get-tuple-element.8)
}
)";

TEST_F(CriticalPathFinderTest, BasicCriticalPathFinderTest) {
  const absl::flat_hash_map<std::string, int64_t> call_to_root_id = {
      {"call", 1},
      {"call.1", 2},
      {"call.2", 0},
  };

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetHloModuleFromText(kBasicHlo, /*num_devices=*/2));

  TF_ASSERT_OK_AND_ASSIGN(auto color_f,
                          partition_->AllocateColor(
                              "task_f", {{.start = 0, .num_devices = 1}}, {}));

  std::vector<HloInstruction*> call_instructions;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kCall) {
      call_instructions.push_back(instruction);
    }
  }

  ASSERT_NE(call_instructions.size(), 0);

  const absl::flat_hash_map<const HloInstruction*, int64_t> depth_map =
      ComputeDepthMap(module->entry_computation());

  for (auto* call_instruction : call_instructions) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto root_id,
        CriticalRootTupleIndex(call_instruction, module->entry_computation(),
                               depth_map));
    EXPECT_EQ(root_id, call_to_root_id.at(call_instruction->name()));
  }
}

}  // namespace
}  // namespace xla
