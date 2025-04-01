#include "xla/pjrt/legate/mpmd_computation_fusion.h"

#include <utility>

#include "gmock/gmock.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdComputationFusionTest : public MpmdTestBase {};

namespace m = ::xla::mpmd_matchers;

using ::testing::Ge;
using ::testing::Lt;

static constexpr absl::string_view kNestedWhileHlo = R"(
HloModule jit_c, num_partitions=2

%task_f_loop (get-tuple-element.4: f32[4,4], get-tuple-element.5: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.4 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %get-tuple-element.5 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %add.0 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.4, f32[4,4]{1,0} %get-tuple-element.5), frontend_attributes={color="task_f"}
  %add.1 = f32[4,4]{1,0} add(f32[4,4]{1,0} %add.0, f32[4,4]{1,0} %get-tuple-element.5), frontend_attributes={color="task_f"}
  ROOT %tuple.3 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.1)
}

%task_g_loop (get-tuple-element.7: f32[4,4], get-tuple-element.8: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.7 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %get-tuple-element.8 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %add.4 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.7, f32[4,4]{1,0} %get-tuple-element.8), frontend_attributes={color="task_f"}
  %add.5 = f32[4,4]{1,0} add(f32[4,4]{1,0} %add.4, f32[4,4]{1,0} %get-tuple-element.8), frontend_attributes={color="task_f"}
  ROOT %tuple.4 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.5)
}

%region_0.83 (arg_tuple.84: (f32[4,4], f32[4,4], f32[4,4], f32[4,4])) -> (f32[4,4], f32[4,4], f32[4,4], f32[4,4]) {
  %arg_tuple.84 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  %get-tuple-element.10 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=0
  %get-tuple-element.11 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=1
  %get-tuple-element.12 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=2
  %call.2 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.10, f32[4,4]{1,0} %get-tuple-element.12), to_apply=%task_f_loop, frontend_attributes={color="task_f"}
  %get-tuple-element.6 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.2), index=0, frontend_attributes={color="task_f"}
  %get-tuple-element.13 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=3
  %call.3 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.6, f32[4,4]{1,0} %get-tuple-element.13), to_apply=%task_g_loop, frontend_attributes={color="task_f"}
  %get-tuple-element.9 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.3), index=0, frontend_attributes={color="task_f"}
  ROOT %tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.10, f32[4,4]{1,0} %get-tuple-element.11, f32[4,4]{1,0} %get-tuple-element.6, f32[4,4]{1,0} %get-tuple-element.9)
}

%region_2.98 (arg_tuple.99: (f32[4,4], f32[4,4], f32[4,4], f32[4,4])) -> pred[] {
  %arg_tuple.99 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  %constant.106 = s32[] constant(2)
  %constant.107 = s32[] constant(2)
  ROOT %compare.108 = pred[] compare(s32[] %constant.106, s32[] %constant.107), direction=LT
}

%task_f () -> (f32[4,4]) {
  %constant.0 = f32[] constant(0), frontend_attributes={color="task_f"}
  %broadcast.1 = f32[4,4]{1,0} broadcast(f32[] %constant.0), dimensions={}, frontend_attributes={color="task_f"}
  %copy.0 = f32[4,4]{1,0} copy(f32[4,4]{1,0} %broadcast.1), frontend_attributes={color="task_f"}
  ROOT %tuple.1 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %copy.0)
}

%task_g () -> (f32[4,4]) {
  %constant.1 = f32[] constant(0), frontend_attributes={color="task_f"}
  %broadcast.2 = f32[4,4]{1,0} broadcast(f32[] %constant.1), dimensions={}, frontend_attributes={color="task_f"}
  %copy.1 = f32[4,4]{1,0} copy(f32[4,4]{1,0} %broadcast.2), frontend_attributes={color="task_f"}
  ROOT %tuple.2 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %copy.1)
}

%task_f.1 (get-tuple-element.14: f32[4,4], Arg_2.0: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.14 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %Arg_2.0 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %add.6 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.14, f32[4,4]{1,0} %Arg_2.0), frontend_attributes={color="task_f"}
  ROOT %tuple.5 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.6)
}

%task_g.1 (get-tuple-element.16: f32[4,4], Arg_3.0: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.16 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %Arg_3.0 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %add.7 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.16, f32[4,4]{1,0} %Arg_3.0), frontend_attributes={color="task_f"}
  ROOT %tuple.6 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.7)
}

ENTRY %main.117 (Arg_0.1: f32[4,4], Arg_1.2: f32[4,4], Arg_2.3: f32[4,4], Arg_3.4: f32[4,4]) -> (f32[4,4], f32[4,4], f32[4,4]) {
  %Arg_0.1 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %Arg_1.2 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %call = (f32[4,4]{1,0}) call(), to_apply=%task_f, frontend_attributes={color="task_f"}
  %get-tuple-element = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call), index=0, frontend_attributes={color="task_f"}
  %call.1 = (f32[4,4]{1,0}) call(), to_apply=%task_g, frontend_attributes={color="task_f"}
  %get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.1), index=0, frontend_attributes={color="task_f"}
  %tuple.10 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %Arg_0.1, f32[4,4]{1,0} %Arg_1.2, f32[4,4]{1,0} %get-tuple-element, f32[4,4]{1,0} %get-tuple-element.1)
  %while.109 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) while((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.10), condition=%region_2.98, body=%region_0.83, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  %get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.109), index=2
  %Arg_2.3 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_f"}
  %call.4 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.2, f32[4,4]{1,0} %Arg_2.3), to_apply=%task_f.1, frontend_attributes={color="task_f"}
  %get-tuple-element.15 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.4), index=0, frontend_attributes={color="task_f"}
  %get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.109), index=3
  %Arg_3.4 = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_f"}
  %call.5 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.15, f32[4,4]{1,0} %Arg_3.4), to_apply=%task_g.1, frontend_attributes={color="task_f"}
  %get-tuple-element.17 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.5), index=0, frontend_attributes={color="task_f"}
  ROOT %tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.15, f32[4,4]{1,0} %get-tuple-element.3, f32[4,4]{1,0} %get-tuple-element.17)
}
)";

TEST_F(MpmdComputationFusionTest, NestedWhileLoop) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kNestedWhileHlo, /*num_devices=*/2));

  const int64_t num_unfused_computations = module->computation_count();

  MpmdComputationFusion fusion{
      partition_.get(), MpmdComputationFusion::FusionType::kMatchingColor,
      /*only_fuse_loop_tasks=*/false};
  HloDCE dce{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(changed, dce.Run(module.get()));

  // fuse before, fuse in the loop, fuse after
  EXPECT_EQ(module->computation_count(), num_unfused_computations - 3);
}

TEST_F(MpmdComputationFusionTest, NestedWhileLoopOnlyFuseLoop) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kNestedWhileHlo, /*num_devices=*/2));

  const int64_t num_unfused_computations = module->computation_count();

  MpmdComputationFusion fusion{
      partition_.get(), MpmdComputationFusion::FusionType::kMatchingColor,
      /*only_fuse_loop_tasks=*/true};
  HloDCE dce{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(changed, dce.Run(module.get()));

  // don't fuse before, fuse in the loop, don't use after
  EXPECT_EQ(module->computation_count(), num_unfused_computations - 1);
}

static constexpr absl::string_view kNestedWhileDifferentColorsInLoopHlo = R"(
HloModule jit_c, num_partitions=2

%task_f_loop (get-tuple-element.4: f32[4,4], get-tuple-element.5: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.4 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %get-tuple-element.5 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %add.0 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.4, f32[4,4]{1,0} %get-tuple-element.5), frontend_attributes={color="task_f"}
  %add.1 = f32[4,4]{1,0} add(f32[4,4]{1,0} %add.0, f32[4,4]{1,0} %get-tuple-element.5), frontend_attributes={color="task_f"}
  ROOT %tuple.3 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.1)
}

%task_g_loop (get-tuple-element.7: f32[4,4], get-tuple-element.8: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.7 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_g"}
  %get-tuple-element.8 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_g"}
  %add.4 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.7, f32[4,4]{1,0} %get-tuple-element.8), frontend_attributes={color="task_g"}
  %add.5 = f32[4,4]{1,0} add(f32[4,4]{1,0} %add.4, f32[4,4]{1,0} %get-tuple-element.8), frontend_attributes={color="task_g"}
  ROOT %tuple.4 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.5)
}

%region_0.83 (arg_tuple.84: (f32[4,4], f32[4,4], f32[4,4], f32[4,4])) -> (f32[4,4], f32[4,4], f32[4,4], f32[4,4]) {
  %arg_tuple.84 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  %get-tuple-element.10 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=0
  %get-tuple-element.11 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=1
  %get-tuple-element.12 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=2
  %call.2 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.10, f32[4,4]{1,0} %get-tuple-element.12), to_apply=%task_f_loop, frontend_attributes={color="task_f"}
  %get-tuple-element.6 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.2), index=0, frontend_attributes={color="task_f"}
  %get-tuple-element.13 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %arg_tuple.84), index=3
  %call.3 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.6, f32[4,4]{1,0} %get-tuple-element.13), to_apply=%task_g_loop, frontend_attributes={color="task_g"}
  %get-tuple-element.9 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.3), index=0, frontend_attributes={color="task_g"}
  ROOT %tuple.97 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.10, f32[4,4]{1,0} %get-tuple-element.11, f32[4,4]{1,0} %get-tuple-element.6, f32[4,4]{1,0} %get-tuple-element.9)
}

%region_2.98 (arg_tuple.99: (f32[4,4], f32[4,4], f32[4,4], f32[4,4])) -> pred[] {
  %arg_tuple.99 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) parameter(0)
  %constant.106 = s32[] constant(2)
  %constant.107 = s32[] constant(2)
  ROOT %compare.108 = pred[] compare(s32[] %constant.106, s32[] %constant.107), direction=LT
}

%task_f () -> (f32[4,4]) {
  %constant.0 = f32[] constant(0), frontend_attributes={color="task_f"}
  %broadcast.1 = f32[4,4]{1,0} broadcast(f32[] %constant.0), dimensions={}, frontend_attributes={color="task_f"}
  %copy.0 = f32[4,4]{1,0} copy(f32[4,4]{1,0} %broadcast.1), frontend_attributes={color="task_f"}
  ROOT %tuple.1 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %copy.0)
}

%task_g () -> (f32[4,4]) {
  %constant.1 = f32[] constant(0), frontend_attributes={color="task_f"}
  %broadcast.2 = f32[4,4]{1,0} broadcast(f32[] %constant.1), dimensions={}, frontend_attributes={color="task_f"}
  %copy.1 = f32[4,4]{1,0} copy(f32[4,4]{1,0} %broadcast.2), frontend_attributes={color="task_f"}
  ROOT %tuple.2 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %copy.1)
}

%task_f.1 (get-tuple-element.14: f32[4,4], Arg_2.0: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.14 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %Arg_2.0 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %add.6 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.14, f32[4,4]{1,0} %Arg_2.0), frontend_attributes={color="task_f"}
  ROOT %tuple.5 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.6)
}

%task_g.1 (get-tuple-element.16: f32[4,4], Arg_3.0: f32[4,4]) -> (f32[4,4]) {
  %get-tuple-element.16 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %Arg_3.0 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %add.7 = f32[4,4]{1,0} add(f32[4,4]{1,0} %get-tuple-element.16, f32[4,4]{1,0} %Arg_3.0), frontend_attributes={color="task_f"}
  ROOT %tuple.6 = (f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %add.7)
}

ENTRY %main.117 (Arg_0.1: f32[4,4], Arg_1.2: f32[4,4], Arg_2.3: f32[4,4], Arg_3.4: f32[4,4]) -> (f32[4,4], f32[4,4], f32[4,4]) {
  %Arg_0.1 = f32[4,4]{1,0} parameter(0), frontend_attributes={color="task_f"}
  %Arg_1.2 = f32[4,4]{1,0} parameter(1), frontend_attributes={color="task_f"}
  %call = (f32[4,4]{1,0}) call(), to_apply=%task_f, frontend_attributes={color="task_f"}
  %get-tuple-element = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call), index=0, frontend_attributes={color="task_f"}
  %call.1 = (f32[4,4]{1,0}) call(), to_apply=%task_g, frontend_attributes={color="task_f"}
  %get-tuple-element.1 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.1), index=0, frontend_attributes={color="task_f"}
  %tuple.10 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %Arg_0.1, f32[4,4]{1,0} %Arg_1.2, f32[4,4]{1,0} %get-tuple-element, f32[4,4]{1,0} %get-tuple-element.1)
  %while.109 = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) while((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %tuple.10), condition=%region_2.98, body=%region_0.83, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  %get-tuple-element.2 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.109), index=2
  %Arg_2.3 = f32[4,4]{1,0} parameter(2), frontend_attributes={color="task_f"}
  %call.4 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.2, f32[4,4]{1,0} %Arg_2.3), to_apply=%task_f.1, frontend_attributes={color="task_f"}
  %get-tuple-element.15 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.4), index=0, frontend_attributes={color="task_f"}
  %get-tuple-element.3 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) %while.109), index=3
  %Arg_3.4 = f32[4,4]{1,0} parameter(3), frontend_attributes={color="task_f"}
  %call.5 = (f32[4,4]{1,0}) call(f32[4,4]{1,0} %get-tuple-element.15, f32[4,4]{1,0} %Arg_3.4), to_apply=%task_g.1, frontend_attributes={color="task_f"}
  %get-tuple-element.17 = f32[4,4]{1,0} get-tuple-element((f32[4,4]{1,0}) %call.5), index=0, frontend_attributes={color="task_f"}
  ROOT %tuple = (f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4,4]{1,0}) tuple(f32[4,4]{1,0} %get-tuple-element.15, f32[4,4]{1,0} %get-tuple-element.3, f32[4,4]{1,0} %get-tuple-element.17)
}
)";

TEST_F(MpmdComputationFusionTest, NestedWhileLoopDeviceFusion) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kNestedWhileDifferentColorsInLoopHlo,
                                        /*num_devices=*/2));

  TF_ASSIGN_OR_RETURN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 0, .num_devices = 2}}, nullptr));
  const int64_t num_unfused_computations = module->computation_count();

  MpmdComputationFusion fusion{
      partition_.get(), MpmdComputationFusion::FusionType::kMatchingDevices,
      /*only_fuse_loop_tasks=*/true};
  HloDCE dce{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(changed, dce.Run(module.get()));

  // don't fuse before, fuse in the loop, don't fuse after
  EXPECT_EQ(module->computation_count(), num_unfused_computations - 1);
}

TEST_F(MpmdComputationFusionTest, NestedWhileLoopMismatchedDeviceNoFusion) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kNestedWhileDifferentColorsInLoopHlo,
                                        /*num_devices=*/2));

  TF_ASSIGN_OR_RETURN(
      auto f,
      partition_->AllocateColor(
          "task_f", zuku::DeviceList{{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto g,
      partition_->AllocateColor(
          "task_g", zuku::DeviceList{{.start = 2, .num_devices = 4}}, nullptr));
  const int64_t num_unfused_computations = module->computation_count();

  MpmdComputationFusion fusion{
      partition_.get(), MpmdComputationFusion::FusionType::kMatchingDevices,
      /*only_fuse_loop_tasks=*/true};
  HloDCE dce{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(changed, dce.Run(module.get()));

  // don't fuse before, don't fuse in the loop, don't fuse after
  EXPECT_EQ(module->computation_count(), num_unfused_computations);
}

static constexpr absl::string_view kOnlyFuseForwardHlo = R"(
red1 {
  param.0 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  add.0 = f32[8]{0} add(param.0, param.0), frontend_attributes={color="red"}
  ROOT tuple.0 = (f32[8]{0}) tuple(add.0)
}

blue1 {
  param.1 = f32[8]{0} parameter(0), frontend_attributes={color="blue"}
  param.2 = f32[8]{0} parameter(1), frontend_attributes={color="blue"}
  add.1 = f32[8]{0} add(param.1, param.2), frontend_attributes={color="blue"}
  ROOT tuple.1 = (f32[8]{0}) tuple(add.1)
}

red2 {
  param.3 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  param.4 = f32[8]{0} parameter(1), frontend_attributes={color="red"}
  add.2 = f32[8]{0} add(param.3, param.4), frontend_attributes={color="red"}
  ROOT tuple.2 = (f32[8]{0}) tuple(add.2)
}

ENTRY main {
  Arg_0.1 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  Arg_1.2 = f32[8]{0} parameter(1), frontend_attributes={color="blue"}
  Arg_2.3 = f32[8]{0} parameter(2), frontend_attributes={color="red"}
  call.0 = (f32[8]{0}) call(Arg_0.1), to_apply=red1, frontend_attributes={color="red"}
  get-tuple-element.0 = f32[8]{0} get-tuple-element(call.0), index=0, frontend_attributes={color="red"}
  call.1 = (f32[8]{0}) call(Arg_1.2, get-tuple-element.0), to_apply=blue1, frontend_attributes={color="blue"}
  get-tuple-element.1 = f32[8]{0} get-tuple-element(call.1), index=0, frontend_attributes={color="blue"}
  call.2 = (f32[8]{0}) call(Arg_2.3, get-tuple-element.0), to_apply=red2, frontend_attributes={color="red"}
  get-tuple-element.2 = f32[8]{0} get-tuple-element(call.2), index=0, frontend_attributes={color="red"}
  ROOT tuple = (f32[8]{0}, f32[8]{0}, f32[8]{0}) tuple(get-tuple-element.0, get-tuple-element.1, get-tuple-element.2)
}
)";

TEST_F(MpmdComputationFusionTest, OnlyFuseForward) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetHloModuleFromText(kOnlyFuseForwardHlo,
                                                            /*num_devices=*/4));

  TF_ASSIGN_OR_RETURN(
      auto red,
      partition_->AllocateColor(
          "red", zuku::DeviceList{{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto blue,
      partition_->AllocateColor(
          "blue", zuku::DeviceList{{.start = 2, .num_devices = 2}}, nullptr));

  MpmdComputationFusion fusion{
      partition_.get(), MpmdComputationFusion::FusionType::kMatchingColor,
      /*only_fuse_loop_tasks=*/false};
  HloDCE dce{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(changed, dce.Run(module.get()));

  // tasks should only fuse later
  EXPECT_THAT(m::EntryComputationCalls(module.get()),
              ElementsAre(m::Color("red"), m::Color("blue"), m::Color("red")));
}

static constexpr absl::string_view kDoNotFuseBackwardsHlo = R"(
red1 {
  param.0 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  add.0 = f32[8]{0} add(param.0, param.0), frontend_attributes={color="red"}
  ROOT tuple.0 = (f32[8]{0}) tuple(add.0)
}

blue1 {
  param.1 = f32[8]{0} parameter(0), frontend_attributes={color="blue"}
  param.2 = f32[8]{0} parameter(1), frontend_attributes={color="blue"}
  add.1 = f32[8]{0} add(param.1, param.2), frontend_attributes={color="blue"}
  ROOT tuple.1 = (f32[8]{0}) tuple(add.1)
}

red2 {
  param.3 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  param.4 = f32[8]{0} parameter(1), frontend_attributes={color="red"}
  add.2 = f32[8]{0} add(param.3, param.4), frontend_attributes={color="red"}
  ROOT tuple.2 = (f32[8]{0}) tuple(add.2)
}

ENTRY main {
  Arg_0.1 = f32[8]{0} parameter(0), frontend_attributes={color="red"}
  Arg_1.2 = f32[8]{0} parameter(1), frontend_attributes={color="blue"}
  Arg_2.3 = f32[8]{0} parameter(2), frontend_attributes={color="red"}
  call.0 = (f32[8]{0}) call(Arg_0.1), to_apply=red1, frontend_attributes={color="red"}
  get-tuple-element.0 = f32[8]{0} get-tuple-element(call.0), index=0, frontend_attributes={color="red"}
  call.1 = (f32[8]{0}) call(Arg_1.2, get-tuple-element.0), to_apply=blue1, frontend_attributes={color="blue"}
  get-tuple-element.1 = f32[8]{0} get-tuple-element(call.1), index=0, frontend_attributes={color="blue"}
  call.2 = (f32[8]{0}) call(Arg_2.3, get-tuple-element.1), to_apply=red2, frontend_attributes={color="red"}
  get-tuple-element.2 = f32[8]{0} get-tuple-element(call.2), index=0, frontend_attributes={color="red"}
  ROOT tuple = (f32[8]{0}, f32[8]{0}, f32[8]{0}) tuple(get-tuple-element.0, get-tuple-element.1, get-tuple-element.2)
}
)";

TEST_F(MpmdComputationFusionTest, DoNotFuseBackwards) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kDoNotFuseBackwardsHlo, /*num_devices=*/4));

  TF_ASSIGN_OR_RETURN(
      auto red,
      partition_->AllocateColor(
          "red", zuku::DeviceList{{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto blue,
      partition_->AllocateColor(
          "blue", zuku::DeviceList{{.start = 2, .num_devices = 2}}, nullptr));

  MpmdComputationFusion fusion{
      partition_.get(), MpmdComputationFusion::FusionType::kMatchingColor,
      /*only_fuse_loop_tasks=*/false};
  HloDCE dce{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(changed, dce.Run(module.get()));

  EXPECT_THAT(m::EntryComputationCalls(module.get()),
              ElementsAre(m::Color("red"), m::Color("blue"), m::Color("red")));
}

TEST_F(MpmdComputationFusionTest, FuseSplitBackpropLayers) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("fuse_split_bwd_layers.txt", /*num_devices=*/4));

  for (int i = 0; i < 24; ++i) {
    std::string color = absl::StrCat("layers_", i);
    int offset = 2 * (i % 2);
    TF_ASSIGN_OR_RETURN(
        auto _, partition_->AllocateColor(
                    std::move(color),
                    zuku::DeviceList{{.start = offset, .num_devices = 2}}));
  }
  TF_ASSIGN_OR_RETURN(
      auto emb,
      partition_->AllocateColor(
          "emb", zuku::DeviceList{{.start = 0, .num_devices = 2}}, nullptr));
  TF_ASSIGN_OR_RETURN(
      auto loss,
      partition_->AllocateColor(
          "compute_loss", zuku::DeviceList{{.start = 2, .num_devices = 2}},
          nullptr));
  TF_ASSIGN_OR_RETURN(
      auto final,
      partition_->AllocateColor(
          "final_ln", zuku::DeviceList{{.start = 2, .num_devices = 2}},
          nullptr));

  const int64_t initial_computations_count = module->computation_count();
  // the module should start with 5 computations each for 24 layers
  EXPECT_THAT(initial_computations_count, Ge(120));

  // only fuse the split tasks inside the loop
  MpmdComputationFusion fusion{partition_.get(),
                               MpmdComputationFusion::kMatchingColor,
                               /*only_fuse_loop_tasks=*/true};
  HloDCE dce{};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, fusion.Run(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(changed, dce.Run(module.get()));

  // each of the layers should have fused 1 together
  EXPECT_THAT(module->computation_count(), Lt(initial_computations_count - 24));
}

}  // namespace
}  // namespace xla
