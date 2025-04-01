#include "xla/pjrt/legate/mpmd_insert_reshard.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::Field;
using ::testing::Not;

class MpmdInsertReshardTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

constexpr absl::string_view kBasicTasksHlo = R"(
task_f {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  param.1 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  add.0 = f32[4,4]{1,0} add(param.0, param.1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  add.1 = f32[4,4]{1,0} add(add.0, param.1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.0, add.1)
}

task_g {
  param.2 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.3 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.4 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.5  = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.4 = f32[4,4]{1,0} add(param.2, param.3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.5 = f32[4,4]{1,0} add(param.4, param.5), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.4 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.4, add.5)
}

task_init {
  zero = f32[] constant(0.0)
  broadcast.0 = f32[4,4]{1,0} broadcast(zero), dimensions={}, frontend_attributes={color="task_f"}
  broadcast.1 = f32[4,4]{1,0} broadcast(zero), dimensions={}, frontend_attributes={color="task_f"}
  ROOT tuple.1 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(broadcast.0, broadcast.1)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  Arg_1.2 = f32[4,4]{1,0} parameter(1), sharding={devices=[4,1]0,1,2,3}
  call.0 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(), to_apply=task_init, frontend_attributes={color="task_f"}
  get-tuple-element.4 = f32[4,4]{1,0} get-tuple-element(call.0), index=0, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  get-tuple-element.5 = f32[4,4]{1,0} get-tuple-element(call.0), index=1, sharding={devices=[4,1]0,1,2,3}, frontend_attributes={color="task_f"}
  call.2 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call.2), index=1, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(get-tuple-element.0, get-tuple-element.1, get-tuple-element.4, get-tuple-element.5), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.3), index=1, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.2, get-tuple-element.3)
}
)";

constexpr absl::string_view k4x1ShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 4
tile_assignment_dimensions: 1
iota_reshape_dims: 4
iota_transpose_perm: 0
)";

TEST_F(MpmdInsertReshardTest, BasicTasks) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kBasicTasksHlo, /*num_devices=*/4));

  TF_ASSIGN_OR_RETURN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 4}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 4, .num_devices = 4}}, nullptr));

  MpmdInsertReshard inserter{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, GetSharding(k4x1ShardingPbtxt));

  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(AllOf(op::CustomCall("Reshard"), op::Sharding(sharding),
                             m::Color("task_g")))
                  .Times(4));
}

constexpr absl::string_view kCommonReshardHlo = R"(
task {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  param.1 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  add.0 = f32[4,4]{1,0} add(param.0, param.1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  add.1 = f32[4,4]{1,0} add(add.0, param.1), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.0, add.1)
}

aggregate {
  param.2 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.3 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.4 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.5  = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.4 = f32[4,4]{1,0} add(param.2, param.3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.5 = f32[4,4]{1,0} add(param.4, param.5), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.4 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.4, add.5)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  Arg_1.2 = f32[4,4]{1,0} parameter(1), sharding={devices=[4,1]4,5,6,7}
  call.2 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2), to_apply=task, frontend_attributes={color="task_g"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element(call.2), index=1, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(Arg_0.1, Arg_1.2, get-tuple-element.0, get-tuple-element.1), to_apply=aggregate, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(call.3), index=1, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(get-tuple-element.2, get-tuple-element.3)
}
)";

TEST_F(MpmdInsertReshardTest, CommonReshard) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kCommonReshardHlo, /*num_devices=*/4));

  TF_ASSIGN_OR_RETURN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 4}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 4, .num_devices = 4}}, nullptr));

  MpmdInsertReshard inserter{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, GetSharding(k4x1ShardingPbtxt));

  // there should be a single reshard used in two places
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(AllOf(op::CustomCall("Reshard"), op::Sharding(sharding),
                             m::Color("task_g"),
                             m::Users(ElementsAre(op::Call(), op::Call()))))
                  .Times(1));
}

constexpr absl::string_view kShardingChangeSameDevicesHlo = R"(
task_f {
  param.0 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}, sharding={replicated}
  add.0 = f32[4,4]{1,0} add(param.0, param.0), frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.3 = (f32[4,4]{1,0}) tuple(add.0)
}

task_g {
  param.2 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  param.3 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  add.4 = f32[4,4]{1,0} add(param.2, param.3), frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.4 = (f32[4,4]{1,0}) tuple(add.4)
}

ENTRY main {
  Arg_0.1 = f32[4,4]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  call.2 = (f32[4,4]{1,0}) call(Arg_0.1), to_apply=task_f, frontend_attributes={color="task_f"}
  get-tuple-element.0 = f32[4,4]{1,0} get-tuple-element(call.2), index=0, frontend_attributes={color="task_f"}, sharding={devices=[4,1]0,1,2,3}
  call.3 = (f32[4,4]{1,0}, f32[4,4]{1,0}) call(get-tuple-element.0, Arg_0.1), to_apply=task_g, frontend_attributes={color="task_g"}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(call.3), index=0, frontend_attributes={color="task_g"}, sharding={devices=[4,1]0,1,2,3}
  ROOT tuple.97 = (f32[4,4]{1,0}) tuple(get-tuple-element.2)
}
)";

TEST_F(MpmdInsertReshardTest, ShardingChangeSameDevices) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath(kShardingChangeSameDevicesHlo, /*num_devices=*/4));

  TF_ASSIGN_OR_RETURN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 4}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 4, .num_devices = 4}}, nullptr));

  MpmdInsertReshard inserter{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inserter.Run(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(HloSharding sharding, GetSharding(k4x1ShardingPbtxt));

  // the parameter should get resharded into both uses
  // the first is over the same devices, but changes from sharded to replicated
  // the second has the same sharding, but different devices
  EXPECT_THAT(
      module->entry_computation()->parameter_instruction(0),
      m::Users(UnorderedElementsAre(
          AllOf(op::CustomCall("Reshard"),
                op::Sharding(HloSharding::Replicate()), m::Color("task_f")),
          AllOf(op::CustomCall("Reshard"), op::Sharding(sharding),
                m::Color("task_g")))));
}

}  // namespace
}  // namespace xla
