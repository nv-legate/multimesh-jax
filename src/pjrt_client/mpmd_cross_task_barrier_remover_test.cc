/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_cross_task_barrier_remover.h"

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdCrossTaskBarrierRemoverTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

static constexpr absl::string_view kTaskWithSplitBarrierHlo = R"(
HloModule module_main.15, entry_computation_layout={(f32[8]{0}, s32[])->f32[]}, num_partitions=2

%region_0.10 (Arg_0.11: f32[], Arg_1.12: f32[]) -> f32[] {
  %Arg_0.11 = f32[] parameter(0)
  %Arg_1.12 = f32[] parameter(1)
  ROOT %add.13 = f32[] add(f32[] %Arg_0.11, f32[] %Arg_1.12), metadata={op_name="jit(c)/jit(main)/task_g/reduce_sum[axes=(0,)]"}
}

%task_f (Arg_0.0: f32[8], Arg_1.0: s32[]) -> (f32[8]) {
  %Arg_0.0 = f32[8]{0} parameter(0), frontend_attributes={color="task_f"}
  %multiply.0 = f32[8]{0} multiply(f32[8]{0} %Arg_0.0, f32[8]{0} %Arg_0.0), frontend_attributes={color="task_f"}
  %Arg_1.0 = s32[] parameter(1), frontend_attributes={color="task_f"}
  %convert.0 = f32[] convert(s32[] %Arg_1.0), frontend_attributes={color="task_f"}
  %broadcast.0 = f32[8]{0} broadcast(f32[] %convert.0), dimensions={}, frontend_attributes={color="task_f"}
  %multiply.1 = f32[8]{0} multiply(f32[8]{0} %multiply.0, f32[8]{0} %broadcast.0), frontend_attributes={color="task_f"}
  ROOT %tuple = (f32[8]{0}) tuple(f32[8]{0} %multiply.1)
}

%task_g (multiply.2: f32[8]) -> (f32[]) {
  %multiply.2 = f32[8]{0} parameter(0), frontend_attributes={color="task_g"}
  %exp.0 = f32[8]{0} exponential(f32[8]{0} %multiply.2), frontend_attributes={color="task_g"}
  %add.0 = f32[8]{0} add(f32[8]{0} %exp.0, f32[8]{0} %multiply.2), frontend_attributes={color="task_g"}
  %tuple.3 = (f32[8]{0}, f32[8]{0}, f32[8]{0}) tuple(f32[8]{0} %multiply.2, f32[8]{0} %exp.0, f32[8]{0} %add.0), frontend_attributes={color="task_g"}
  %opt-barrier.0 = (f32[8]{0}, f32[8]{0}, f32[8]{0}) opt-barrier((f32[8]{0}, f32[8]{0}, f32[8]{0}) %tuple.3), frontend_attributes={color="task_g"}
  %gte.exp.0 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[8]{0}, f32[8]{0}) %opt-barrier.0), index=1, frontend_attributes={color="task_g"}
  %gte.add.0 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[8]{0}, f32[8]{0}) %opt-barrier.0), index=2, frontend_attributes={color="task_g"}
  %add.1 = f32[8]{0} add(f32[8]{0} %gte.exp.0, f32[8]{0} %gte.add.0), frontend_attributes={color="task_g"}
  %tuple.4 = (f32[8]{0}, f32[8]{0}) tuple(f32[8]{0} %add.1, f32[8]{0} %multiply.2), frontend_attributes={color="task_g"}
  %opt-barrier.3 = (f32[8]{0}, f32[8]{0}) opt-barrier((f32[8]{0}, f32[8]{0}) %tuple.4), frontend_attributes={color="task_g"}
  %gte.add.3 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[8]{0}) %opt-barrier.3), index=0, frontend_attributes={color="task_g"}
  %gte.exp.3 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[8]{0}) %opt-barrier.3), index=1, frontend_attributes={color="task_g"}
  %add.2 = f32[] add(f32[8]{0} %gte.exp.3, f32[8]{0} %gte.add.3), frontend_attributes={color="task_g"}
  ROOT %tuple.5 = (f32[]) tuple(f32[] %add.2)
}

ENTRY %main.15 (Arg_0.1: f32[8], Arg_1.2: s32[]) -> f32[] {
  %Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}, frontend_attributes={color="task_f"}
  %Arg_1.2 = s32[] parameter(1), sharding={replicated}, frontend_attributes={color="task_f"}
  %call = (f32[8]{0}) call(f32[8]{0} %Arg_0.1, s32[] %Arg_1.2), to_apply=%task_f, frontend_attributes={color="task_f"}
  %get-tuple-element = f32[8]{0} get-tuple-element((f32[8]{0}) %call), index=0, frontend_attributes={color="task_f"}
  %call.1 = (f32[]) call(f32[8]{0} %get-tuple-element), to_apply=%task_g, frontend_attributes={color="task_g"}
  ROOT %get-tuple-element.1 = f32[] get-tuple-element((f32[]) %call.1), index=0, frontend_attributes={color="task_g"}
}
)";

TEST_F(MpmdCrossTaskBarrierRemoverTest, TaskWithBarrier) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kTaskWithSplitBarrierHlo, /*num_devices=*/4));

  MpmdCrossTaskBarrierRemover remover{partition_.get(),
                                      /*remove_parameters=*/true};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, remover.Run(module.get()));

  auto* task_g_comp = module->entry_computation()
                          ->GetInstructionWithName("call.1")
                          ->called_computations()[0];

  auto* exp = task_g_comp->GetInstructionWithName("exp.0");
  auto* add = task_g_comp->GetInstructionWithName("add.0");

  // there should be a new subset tuple with two elements with color g
  // the second optimization barrier should be removed since it the paramater
  // is peeled off leaving a single operand
  EXPECT_THAT(task_g_comp->instructions(),
              AllOf(Contains(AllOf(op::Tuple(exp, add), m::Color("task_g"))),
                    Contains(op::OptimizationBarrier()).Times(1)));
}

}  // namespace
}  // namespace xla
