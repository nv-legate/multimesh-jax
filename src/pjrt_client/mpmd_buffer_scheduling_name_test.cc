/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_buffer_scheduling_name.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::Gt;
using ::testing::Not;

class MpmdAssignBufferSchedulingNameTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

constexpr absl::string_view kBasicTasksHlo = R"(
task_f {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  add.0 = f32[4,4]{1,0} add(param.0, param.0), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.3 = (f32[4,4]{1,0}) tuple(add.0)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  call.0 = (f32[4,4]{1,0}) call(Arg_0.1), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.0), index=0, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  call.1 = (f32[4,4]{1,0}) call(get-tuple-element.0), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call.1), index=0, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  call.2 = (f32[4,4]{1,0}) call(get-tuple-element.1), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  call.3 = (f32[4,4]{1,0}) call(get-tuple-element.2), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  call.4 = (f32[4,4]{1,0}) call(get-tuple-element.3), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.4 = f32[4,4]{1,0} get-tuple-element(call.4), index=0, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.97 = (f32[4,4]{1,0}) tuple(get-tuple-element.4)
}
)";

TEST_F(MpmdAssignBufferSchedulingNameTest, BasicTasks) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kBasicTasksHlo, /*num_devices=*/4));

  MpmdAssignBufferSchedulingName assigner{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, assigner.Run(module.get()));

  // all get-tuple-element ops should have been assigned scheduling names
  EXPECT_THAT(module->entry_computation()->instructions(),
              Not(Contains(AllOf(op::GetTupleElement(),
                                 m::MetadataSchedulingNames("")))));

  // the scheduling name should be assigned and used more than once
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(AllOf(op::GetTupleElement(),
                             m::MetadataSchedulingNames("get-tuple-element.0")))
                  .Times(Gt(1)));
}

constexpr absl::string_view kScheduleReshardHlo = R"(
task_init {
  zero = f32[] constant(0)
  broadcast.0 = f32[4,4]{1,0} broadcast(zero), dimensions={}, frontend_attributes={color="task_f"}
  broadcast.1 = f32[4,4]{1,0} broadcast(zero), dimensions={}, frontend_attributes={color="task_f"}
  ROOT tuple.1 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(broadcast.0, broadcast.1)
}

task_f {
  param.0 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  param.1 = f32[4,4]{1,0} parameter(1), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  add.0 = f32[4,4]{1,0} add(param.0, param.1), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  add.1 = f32[4,4]{1,0} add(add.0, param.1), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.0, add.1)
}

task_g {
  param.2 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  param.3 = f32[4,4]{1,0} parameter(1), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  add.4 = f32[4,4]{1,0} add(param.2, param.3), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  param.4 = f32[4,4]{1,0} parameter(2), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  param.5 = f32[4,4]{1,0} parameter(3), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  add.5 = f32[4,4]{1,0} add(param.4, param.5), sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  ROOT tuple.4 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.4, add.5)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  Arg_1.2 = f32[4,4]{1,0} parameter(1), sharding={devices=[4,1]0,1,2,3}
  call.0 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(), to_apply=task_init, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.0), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  get-tuple-element.10 = f32[4,4]{1,0} get-tuple-element(call.0), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  custom-call.0 = f32[4,4]{1,0} custom-call(get-tuple-element.0), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.4"}
  custom-call.10 = f32[4,4]{1,0} custom-call(get-tuple-element.10), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.41"}
  call.1 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call.1), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  get-tuple-element.11 = f32[4,4]{1,0} get-tuple-element(call.1), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  custom-call.1 = f32[4,4]{1,0} custom-call(get-tuple-element.1), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  custom-call.11 = f32[4,4]{1,0} custom-call(get-tuple-element.11), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  call.4 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(custom-call.0, custom-call.10, custom-call.1, custom-call.11), to_apply=task_g, frontend_attributes={color="task_g"}, control-predecessors={call.1}
  get-tuple-element.4 = f32[4,4]{1,0} get-tuple-element(call.4), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.4"}
  get-tuple-element.41 = f32[4,4]{1,0} get-tuple-element(call.4), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.41"}
  call.2 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task_f, frontend_attributes={color="task_f"}, control-predecessors={call.4}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  custom-call.2 = f32[4,4]{1,0} custom-call(get-tuple-element.2), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  get-tuple-element.21 = f32[4,4]{1,0} get-tuple-element(call.2), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  custom-call.21 = f32[4,4]{1,0} custom-call(get-tuple-element.2), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  call.5 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(custom-call.0, custom-call.10, get-tuple-element.4, get-tuple-element.41), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.5 = f32[4,4]{1,0} get-tuple-element(call.5), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.4"}
  get-tuple-element.51 = f32[4,4]{1,0} get-tuple-element(call.5), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.41"}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task_f, frontend_attributes={color="task_f"}, control-predecessors={call.5}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  custom-call.3 = f32[4,4]{1,0} custom-call(get-tuple-element.3), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  get-tuple-element.31 = f32[4,4]{1,0} get-tuple-element(call.3), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  custom-call.31 = f32[4,4]{1,0} custom-call(get-tuple-element.3), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  call.6 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(custom-call.0, custom-call.10, get-tuple-element.5, get-tuple-element.51), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.6 = f32[4,4]{1,0} get-tuple-element(call.6), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.4"}
  get-tuple-element.61 = f32[4,4]{1,0} get-tuple-element(call.6), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.41"}
  call.7 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task_f, frontend_attributes={color="task_f"}, control-predecessors={call.6}
  get-tuple-element.7 = f32[4,4]{1,0} get-tuple-element(call.7), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  custom-call.7 = f32[4,4]{1,0} custom-call(get-tuple-element.7), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  get-tuple-element.71 = f32[4,4]{1,0} get-tuple-element(call.7), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  custom-call.71 = f32[4,4]{1,0} custom-call(get-tuple-element.7), custom_call_target="Reshard", sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}
  call.8 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(custom-call.0, custom-call.10, get-tuple-element.6, get-tuple-element.61), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.8 = f32[4,4]{1,0} get-tuple-element(call.8), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.4"}
  get-tuple-element.81 = f32[4,4]{1,0} get-tuple-element(call.8), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_g"}, metadata={scheduling_name="get-tuple-element.41"}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.8, get-tuple-element.81)
}
)";

TEST_F(MpmdAssignBufferSchedulingNameTest, ScheduleReshard) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetHloModuleFromText(kScheduleReshardHlo,
                                                            /*num_devices=*/8));

  TF_ASSIGN_OR_RETURN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 4}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 4, .num_devices = 4}}, nullptr));

  MpmdAssignBufferSchedulingName assigner{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, assigner.Run(module.get()));

  // every task output or reshard op should have a buffer name assigned
  EXPECT_THAT(module->entry_computation()->instructions(),
              Not(Contains(
                  AllOf(AnyOf(op::GetTupleElement(), op::CustomCall("Reshard")),
                        m::MetadataSchedulingNames("")))));

  // a variable that has been explicitly assigned a scheduling name should not
  // be changed or re-used, 5 = init + 4 updates
  EXPECT_THAT(
      module->entry_computation()->instructions(),
      Contains(m::MetadataSchedulingNames("get-tuple-element.4")).Times(5));

  // the output of the reshard should be reused
  EXPECT_THAT(
      module->entry_computation()->instructions(),
      Contains(m::MetadataSchedulingNames("custom-call.1")).Times(Gt(1)));
}

TEST_F(MpmdAssignBufferSchedulingNameTest, AliasBufferAssignment) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("alias-buffer-assignment.txt", /*num_devices=*/8));

  TF_ASSIGN_OR_RETURN(
      auto emb,
      partition_->AllocateColor(
          "emb", zuku::DeviceList{{.start = 0, .num_devices = 4}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto decoder_norm,
      partition_->AllocateColor(
          "decoder_norm", zuku::DeviceList{{.start = 4, .num_devices = 4}},
          nullptr));
  TF_ASSIGN_OR_RETURN(
      auto layers_0,
      partition_->AllocateColor(
          "layers_0", zuku::DeviceList{{.start = 0, .num_devices = 4}},
          nullptr));
  TF_ASSIGN_OR_RETURN(
      auto layers_1,
      partition_->AllocateColor(
          "layers_1", zuku::DeviceList{{.start = 4, .num_devices = 4}},
          nullptr));
  TF_ASSIGN_OR_RETURN(
      auto layers_2,
      partition_->AllocateColor(
          "layers_2", zuku::DeviceList{{.start = 0, .num_devices = 4}},
          nullptr));
  TF_ASSIGN_OR_RETURN(
      auto layers_3,
      partition_->AllocateColor(
          "layers_3", zuku::DeviceList{{.start = 4, .num_devices = 4}},
          nullptr));
  TF_ASSIGN_OR_RETURN(
      auto layers_4,
      partition_->AllocateColor(
          "layers_4", zuku::DeviceList{{.start = 0, .num_devices = 4}},
          nullptr));
  TF_ASSIGN_OR_RETURN(
      auto layers_5,
      partition_->AllocateColor(
          "layers_5", zuku::DeviceList{{.start = 4, .num_devices = 4}},
          nullptr));
  TF_ASSIGN_OR_RETURN(
      auto layers_6,
      partition_->AllocateColor(
          "layers_6", zuku::DeviceList{{.start = 0, .num_devices = 4}},
          nullptr));
  TF_ASSIGN_OR_RETURN(
      auto layers_7,
      partition_->AllocateColor(
          "layers_7", zuku::DeviceList{{.start = 4, .num_devices = 4}},
          nullptr));

  MpmdAssignBufferSchedulingName assigner{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, assigner.Run(module.get()));

  // TODO: need a matcher to verify that a buffer name is not used twice in the
  // same call
}

constexpr absl::string_view kSubsetDeviceSkipReshardHlo = R"(
task_g {
  param.0 = f32[] parameter(0), frontend_attributes={color="task_g"}, sharding={replicated}
  add.0 = f32[] add(param.0, param.0), frontend_attributes={color="task_g"}, sharding={replicated}
  ROOT tuple.3 = (f32[4,4]{1,0}) tuple(add.0)
}

task_f {
  param.0 = f32[] parameter(0), frontend_attributes={color="task_f"}, sharding={replicated}
  add.0 = f32[] add(param.0, param.0), frontend_attributes={color="task_f"}, sharding={replicated}
  ROOT tuple.3 = (f32[4,4]{1,0}) tuple(add.0)
}

ENTRY main {
  Arg_0.1 = f32[] parameter(0), sharding={replicated}
  call.0 = (f32[]) call(Arg_0.1), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[] get-tuple-element(call.0), index=0, frontend_attributes={color="task_f"}, sharding={replicated}
  call.1 = (f32[]) call(get-tuple-element.0), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.1 = f32[] get-tuple-element(call.1), index=0, frontend_attributes={color="task_g"}, sharding={replicated}
  call.2 = (f32[]) call(get-tuple-element.1), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[] get-tuple-element(call.2), index=0, frontend_attributes={color="task_g"}, sharding={replicated}
  call.3 = (f32[]) call(get-tuple-element.2), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.3 = f32[] get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}, sharding={replicated}
  ROOT tuple.97 = (f32[]) tuple(get-tuple-element.3)
}
)";

TEST_F(MpmdAssignBufferSchedulingNameTest, SubsetDeviceSkipReshard) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetHloModuleFromText(kSubsetDeviceSkipReshardHlo,
                                               /*num_devices=*/8));

  TF_ASSIGN_OR_RETURN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 8}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 4, .num_devices = 4}}, nullptr));

  MpmdAssignBufferSchedulingName assigner{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, assigner.Run(module.get()));

  // we have a replicated array on color task_f sharded over a superset of the
  // devices in task_g, which means we can directly use the array as input
  // without resharding make sure that array is not "recolored" to task_g when
  // freed
  auto* first_output = module->entry_computation()->GetInstructionWithName(
      "get-tuple-element.0");
  ASSERT_NE(first_output, nullptr);
  auto name = first_output->metadata().scheduling_name();
  EXPECT_NE(name, "");
  EXPECT_THAT(module->entry_computation()->instructions(),
              Not(Contains(AllOf(op::GetTupleElement(), m::Color("task_g"),
                                 m::MetadataSchedulingNames("name")))));
}

}  // namespace
}  // namespace xla
