#include "xla/pjrt/legate/mpmd_computation_grouper.h"

#include "gmock/gmock.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_coloring.h"
#include "xla/pjrt/legate/mpmd_computation_inliner.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdComputationGrouperTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

using ::testing::Ge;

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

TEST_F(MpmdComputationGrouperTest, SimpleImplicitTask) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kSimpleImplicitTaskHlo, /*num_devices=*/4));

  RegisterMatcherTestTask("(task_f)", {0, 1}, {2}, {"x"}, {{"x", "batch"}});
  RegisterMatcherTestTask("(task_g)", {2, 3}, {2}, {"x"}, {{"x", "batch"}});

  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(changed, grouper.Run(module.get()));

  // This should have been grouped into 2 tasks
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(op::Call()).Times(2));

  // there should should not be any non-trivial instructions
  // in the entry computation
  EXPECT_THAT(module->entry_computation()->instructions(),
              Each(AnyOf(op::Parameter(), op::GetTupleElement(), op::Call(),
                         op::Tuple())));

  // verify "idempotency" or running inliner-grouper
  MpmdComputationInliner inliner{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(changed, inliner.Run(module.get()));

  TF_ASSERT_OK_AND_ASSIGN(changed, grouper.Run(module.get()));

  // This should have been grouped back into 2 tasks
  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(op::Call()).Times(2));

  // there should should not be any non-trivial instructions
  // in the entry computation
  EXPECT_THAT(module->entry_computation()->instructions(),
              Each(AnyOf(op::Parameter(), op::GetTupleElement(), op::Call(),
                         op::Tuple())));
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
  cosine.40 = f32[8]{0} cosine(Arg_0.37), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/cos"}
  Arg_1.38 = f32[8]{0} parameter(1), metadata={op_name="jit(c)/jit(main)/legate_task"}
  add.41 = f32[8]{0} add(cosine.40, Arg_1.38), metadata={op_name="jit(c)/jit(main)/jvp(jit(g.impl))/add"}
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
  custom-call.65 = (f32[8]{0}, f32[8]{0}) custom-call(custom-call.19, Arg_0.1, constant.3), custom_call_target="LegateTask", called_computations={f_bwd.impl.56}, backend_config={"name": "g", "devices": [1], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  get-tuple-element.67 = f32[8]{0} get-tuple-element(custom-call.65), index=1
  get-tuple-element.66 = f32[8]{0} get-tuple-element(custom-call.65), index=0
  custom-call.95 = (f32[8]{0}, pred[]) custom-call(Arg_0.1, Arg_1.2, get-tuple-element.66), custom_call_target="LegateTask", called_computations={f_bwd.impl_0.83}, backend_config={"name": "f", "devices": [0], "autosharding": {"dims": [1], "device_axes": [], "logical_axes": []}}
  get-tuple-element.96 = f32[8]{0} get-tuple-element(custom-call.95), index=0
  add.98 = f32[8]{0} add(get-tuple-element.67, get-tuple-element.96)
  ROOT tuple.99 = (f32[], f32[8]{0}) tuple(custom-call.43, add.98)
} // main.100
)";

TEST_F(MpmdComputationGrouperTest, MpmdTask) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kMpmdTaskHlo, /*num_devices=*/2));

  MpmdColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, coloring.Run(module.get()));

  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(changed, grouper.Run(module.get()));

  // there should should not be any non-trivial instructions
  // in the entry computation
  EXPECT_THAT(module->entry_computation()->instructions(),
              Each(AnyOf(op::Parameter(), op::GetTupleElement(), op::Call(),
                         op::Tuple())));

  EXPECT_THAT(module->entry_computation()->instructions(),
              Contains(op::Call()).Times(Ge(4)));

  // there should should not be any non-trivial instructions
  // in the entry computation
  EXPECT_THAT(module->entry_computation()->instructions(),
              Each(AnyOf(op::Parameter(), op::GetTupleElement(), op::Call(),
                         op::Tuple())));
}

static constexpr absl::string_view kBasicColoring = R"(
ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0), frontend_attributes={color="task_f"}
  multiply.4 = f32[8]{0} multiply(Arg_0.1, Arg_0.1), frontend_attributes={color="task_f"}
  Arg_1.2 = s32[] parameter(1), frontend_attributes={color="task_f"}
  convert.5 = f32[] convert(Arg_1.2), frontend_attributes={color="task_f"}
  broadcast.6 = f32[8]{0} broadcast(convert.5), dimensions={}
  add.7 = f32[8]{0} add(multiply.4, broadcast.6), frontend_attributes={color="task_f"}
  cosine.8 = f32[8]{0} cosine(add.7), frontend_attributes={color="task_f"}
  add.9 = f32[8]{0} add(cosine.8, add.7), frontend_attributes={color="task_g"}
  ROOT add.10 = f32[8]{0} add(add.9, Arg_0.1), frontend_attributes={color="task_g"}
} // main.15
)";

TEST_F(MpmdComputationGrouperTest, BasicColoring) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kBasicColoring, /*num_devices=*/2));

  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, grouper.Run(module.get()));

  // anything that is not a call should be a trivial op
  EXPECT_THAT(module->entry_computation()->instructions(),
              AllOf(Contains(op::Call()).Times(2),
                    Contains(AllOf(op::Call(), m::Color("task_f"))).Times(1),
                    Contains(AllOf(op::Call(), m::Color("task_g"))).Times(1),
                    Each(AnyOf(m::TrivialOp(), op::Call()))));
}

static constexpr absl::string_view kNestedWhileHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4]{0}, f32[4]{0})->f32[]}

region_0.83 {
  arg_tuple.84 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  get-tuple-element.10 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=0
  get-tuple-element.11 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=1
  get-tuple-element.12 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=2
  get-tuple-element.13 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=3
  add.10 = f32[4,4]{1,0} add(get-tuple-element.10, get-tuple-element.12), frontend_attributes={color="task_f"}
  add.11 = f32[4,4]{1,0} add(get-tuple-element.11, get-tuple-element.13), frontend_attributes={color="task_g"}
  add.12 = f32[4,4]{1,0} add(add.10, get-tuple-element.12), frontend_attributes={color="task_f"}
  add.13 = f32[4,4]{1,0} add(add.11, get-tuple-element.13), frontend_attributes={color="task_g"}
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
  broadcast = f32[4,4] broadcast(constant.6), dimensions={}
  copy.6 = f32[4,4] copy(broadcast), frontend_attributes={color="task_f"}
  copy.7 = f32[4,4] copy(broadcast), frontend_attributes={color="task_g"}
  Arg_0.1 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  Arg_1.2 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}
  Arg_2.3 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_f"}
  Arg_3.4 = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_g"}
  tuple.10 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(Arg_0.1, Arg_1.2, copy.6, copy.7)
  while.109 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) while(tuple.10), condition=region_2.98, body=region_0.83, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element(while.109), index=2
  get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element(while.109), index=3
  add.2 = f32[4,4]{1,0} add(get-tuple-element.2, Arg_2.3), frontend_attributes={color="task_f"}
  add.3 = f32[4,4]{1,0} add(get-tuple-element.3, Arg_3.4), frontend_attributes={color="task_g"}
  ROOT tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(add.2, add.3)
} // main.117
)";

TEST_F(MpmdComputationGrouperTest, NestedWhileColoring) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kNestedWhileHlo, /*num_devices=*/2));

  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, grouper.Run(module.get()));

  EXPECT_THAT(module->entry_computation()->instructions(),
              Each(AnyOf(m::TrivialOp(), op::Call(), op::While())));

  auto* while_instruction =
      module->entry_computation()->GetInstructionWithName("while.109");
  ASSERT_NE(while_instruction, nullptr);

  EXPECT_THAT(while_instruction->called_computations()[0]->instructions(),
              Each(AnyOf(m::TrivialOp(), op::Call())));

  EXPECT_THAT(m::FlatInstructions(module.get()),
              AllOf(Contains(AllOf(op::Add(), m::Color("task_f"))).Times(3),
                    Contains(AllOf(op::Add(), m::Color("task_g"))).Times(3),
                    Contains(AllOf(op::Call(), m::Color("task_f"))).Times(3),
                    Contains(AllOf(op::Call(), m::Color("task_g"))).Times(3)));
}

static constexpr absl::string_view kSplitOptBarrierHlo = R"(
HloModule module_main.15, entry_computation_layout={(f32[8]{0}, s32[])->f32[]}, num_partitions=2

ENTRY %main.15 (Arg_0.1: f32[8], Arg_1.2: s32[]) -> f32[] {
  %Arg_0.1 = f32[8]{0} parameter(0), sharding={replicated}, frontend_attributes={color="task_f"}
  %multiply.4 = f32[8]{0} multiply(f32[8]{0} %Arg_0.1, f32[8]{0} %Arg_0.1), frontend_attributes={color="task_f"}
  %Arg_1.2 = s32[] parameter(1), sharding={replicated}, frontend_attributes={color="task_f"}
  %convert.5 = f32[] convert(s32[] %Arg_1.2), frontend_attributes={color="task_f"}
  %broadcast.6 = f32[8]{0} broadcast(f32[] %convert.5), dimensions={}, frontend_attributes={color="task_f"}
  %multiply.7 = f32[8]{0} multiply(f32[8]{0} %multiply.4, f32[8]{0} %broadcast.6), frontend_attributes={color="task_f"}
  %exp.8 = f32[8]{0} exponential(f32[8]{0} %multiply.7), frontend_attributes={color="task_g"}
  %add.9 = f32[8]{0} add(f32[8]{0} %exp.8, f32[8]{0} %multiply.7), frontend_attributes={color="task_g"}
  %tuple.1 = (f32[8]{0}, f32[8]{0}, f32[8]{0}) tuple(f32[8]{0} %multiply.7, f32[8]{0} %exp.8, f32[8]{0} %add.9)
  %opt-barrier.1 = (f32[8]{0}, f32[8]{0}, f32[8]{0}) opt-barrier((f32[8]{0}, f32[8]{0}, f32[8]{0}) %tuple.1)
  %gte.add.9 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[8]{0}, f32[8]{0}) %opt-barrier.1), index=2, frontend_attributes={color="task_g"}
  %gte.exp.8 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[8]{0}, f32[8]{0}) %opt-barrier.1), index=1, frontend_attributes={color="task_g"}
  %gte.multiply.7 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[8]{0}, f32[8]{0}) %opt-barrier.1), index=0, frontend_attributes={color="task_f"}
  %add.10 = f32[8]{0} add(f32[8]{0} %gte.exp.8, f32[8]{0} %gte.multiply.7), frontend_attributes={color="task_g"}
  %tuple.2 = (f32[8]{0}, f32[8]{0}) tuple(f32[8]{0} %add.10, f32[8]{0} %exp.8), frontend_attributes={color="task_g"}
  %opt-barrier.2 = (f32[8]{0}, f32[8]{0}) opt-barrier((f32[8]{0}, f32[8]{0}) %tuple.2), frontend_attributes={color="task_g"}
  %gte.add.10 = f32[8]{0} get-tuple-element((f32[8]{0}, f32[8]{0}) %opt-barrier.2), index=0, frontend_attributes={color="task_g"}
  ROOT %add.11 = f32[] add(f32[8]{0} %gte.exp.8, f32[8]{0} %gte.add.10), frontend_attributes={color="task_g"}
}
)";

TEST_F(MpmdComputationGrouperTest, SplitOptBarrier) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetHloModuleFromText(kSplitOptBarrierHlo,
                                                            /*num_devices=*/2));

  for (auto* computation : module->computations()) {
    for (auto* instruction : computation->instructions()) {
      instruction->set_metadata_op_name(std::string(instruction->name()));
    }
  }

  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, grouper.Run(module.get()));

  EXPECT_THAT(module->entry_computation()->instructions(),
              Not(Contains(op::OptimizationBarrier())));

  EXPECT_THAT(
      m::CalledComputationInstructions(module.get()),
      AllOf(Contains(Contains(op::OptimizationBarrier()).Times(2)),
            Contains(Not(Contains(op::OptimizationBarrier()))).Times(Ge(1))));
}

TEST_F(MpmdComputationGrouperTest, BadGrouping) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetHloModuleFromPath("bad_grouping.txt",
                                                            /*num_devices=*/1));

  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, grouper.Run(module.get()));
}

static constexpr absl::string_view kMoveAsLateAsPossibleHlo = R"(
ENTRY main.15 {
  Arg_0.1 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  Arg_1.2 = f32[8]{0} parameter(1), frontend_attributes={color="blue"}
  constant.0 = f32[] constant(0.0)
  broadcast.0 = f32[8]{0} broadcast(constant.0), dimensions={}
  copy.0 = f32[8]{0} copy(broadcast.0), frontend_attributes={color="red"}
  broadcast.1 = f32[8]{0} broadcast(constant.0), dimensions={}
  copy.1 = f32[8]{0} copy(broadcast.0), frontend_attributes={color="blue"}
  iota.0 = f32[8]{0} iota(), iota_dimension=0, frontend_attributes={color="red"}
  iota.1 = f32[8]{0} iota(), iota_dimension=0, frontend_attributes={color="blue"}
  add.0 = f32[8]{0}add(Arg_0.1, Arg_0.1), frontend_attributes={color="red"}
  multiply.0 = f32[8]{0} multiply(add.0, Arg_0.1), frontend_attributes={color="red"}
  add.1 = f32[8]{0}add(Arg_1.2, multiply.0), frontend_attributes={color="blue"}
  multiply.1 = f32[8]{0} multiply(add.1, Arg_1.2), frontend_attributes={color="blue"}
  add.2 = f32[8]{0}add(multiply.1, iota.0), frontend_attributes={color="red"}
  add.3 = f32[8]{0}add(add.2, copy.0), frontend_attributes={color="red"}
  add.4 = f32[8]{0} add(multiply.1, iota.1), frontend_attributes={color="blue"}
  add.5 = f32[8]{0} add(add.4, copy.1), frontend_attributes={color="blue"}
  ROOT tuple = (f32[8]{0}, f32[8]{0}) tuple(add.3,add.5)
} // main.15
)";

TEST_F(MpmdComputationGrouperTest, MoveAsLateAsPossible) {
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          GetHloModuleFromText(kMoveAsLateAsPossibleHlo,
                                               /*num_devices=*/1));

  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, grouper.Run(module.get()));

  // the iota should have been moved later to co-locate
  // with its later users
  EXPECT_THAT(m::CalledComputationInstructions(module.get()),
              ElementsAre(Not(Contains(op::Iota())), Contains(op::Iota()),
                          Contains(op::Iota())));
}

TEST_F(MpmdComputationGrouperTest, MoveAsLateAsPossible24Layers) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromPath("move_to_users_grouping_24layers.txt",
                                        /*num_devices=*/16));

  for (auto* computation : module->computations()) {
    for (auto* instruction : computation->instructions()) {
      instruction->set_metadata_op_name(std::string(instruction->name()));
    }
  }

  MpmdComputationGrouper grouper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, grouper.Run(module.get()));

  ShapeProto proto;
  proto.set_element_type(PrimitiveType::F32);
  proto.add_dimensions(49152);
  proto.add_dimensions(12288);
  Shape shape{proto};

  // 24 layers have the parameter x5 variants
  // the gradient zero initialization
  // the gradient output from the while loop
  // the new parameters
  // the new optimizer states (2x)
  // very important that the intermediate optimizer state
  // doesn't leak out as get-tuple-element intermediates
  EXPECT_THAT(
      module->entry_computation()->instructions(),
      Contains(AllOf(op::GetTupleElement(), op::Shape(shape))).Times(120));
}

}  // namespace
}  // namespace xla
