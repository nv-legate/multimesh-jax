#include "xla/pjrt/legate/mpmd_shard_map_loop_reduce.h"

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

using ::testing::AllOf;

class MpmdShardMapReduceTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

constexpr absl::string_view kSimpleShardedConstractionHlo = R"(
region_1.19 {
  Arg_0.20 = s32[] parameter(0)
  Arg_1.21 = s32[] parameter(1)
  ROOT add.22 = s32[] add(Arg_0.20, Arg_1.21)
}

region_0.23 {
  arg_tuple.24 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) parameter(0)
  get-tuple-element.25 = s32[] get-tuple-element(arg_tuple.24), index=0
  constant.31 = s32[] constant(1)
  add.61 = s32[] add(get-tuple-element.25, constant.31)
  get-tuple-element.26 = s32[] get-tuple-element(arg_tuple.24), index=1
  constant.32 = s32[] constant(2)
  add.50 = s32[] add(get-tuple-element.26, constant.32)
  get-tuple-element.27 = s32[4]{0} get-tuple-element(arg_tuple.24), index=2, sharding={replicated}
  get-tuple-element.28 = s32[4,4]{1,0} get-tuple-element(arg_tuple.24), index=3, sharding={devices=[2,1]<=[2]}
  constant.34 = s32[] constant(0)
  compare.35 = pred[] compare(get-tuple-element.26, constant.34), direction=LT
  constant.33 = s32[] constant(4)
  add.36 = s32[] add(get-tuple-element.26, constant.33)
  select.37 = s32[] select(compare.35, add.36, get-tuple-element.26)
  dynamic-slice.38 = s32[2,4]{1,0} dynamic-slice(get-tuple-element.28, select.37, constant.34), dynamic_slice_sizes={2,4}, sharding={devices=[2,1]<=[2]}
  get-tuple-element.29 = s32[4,1]{1,0} get-tuple-element(arg_tuple.24), index=4, sharding={devices=[2,1]<=[2]}
  compare.40 = pred[] compare(get-tuple-element.26, constant.34), direction=LT
  add.41 = s32[] add(get-tuple-element.26, constant.33)
  select.42 = s32[] select(compare.40, add.41, get-tuple-element.26)
  dynamic-slice.43 = s32[2,1]{1,0} dynamic-slice(get-tuple-element.29, select.42, constant.34), dynamic_slice_sizes={2,1}, sharding={devices=[2,1]<=[2]}
  broadcast.51 = s32[2,1]{1,0} broadcast(dynamic-slice.43), dimensions={0,1}
  reshape.52 = s32[2]{0} reshape(broadcast.51)
  broadcast.53 = s32[2,4]{1,0} broadcast(reshape.52), dimensions={0}, sharding={devices=[2,1]<=[2]}
  add.54 = s32[2,4]{1,0} add(dynamic-slice.38, broadcast.53), sharding={devices=[2,1]<=[2]}
  get-tuple-element.30 = s32[4,1]{1,0} get-tuple-element(arg_tuple.24), index=5, sharding={devices=[2,1]<=[2]}
  compare.45 = pred[] compare(get-tuple-element.26, constant.34), direction=LT
  add.46 = s32[] add(get-tuple-element.26, constant.33)
  select.47 = s32[] select(compare.45, add.46, get-tuple-element.26)
  dynamic-slice.48 = s32[2,1]{1,0} dynamic-slice(get-tuple-element.30, select.47, constant.34), dynamic_slice_sizes={2,1}, sharding={devices=[2,1]<=[2]}
  broadcast.55 = s32[2,1]{1,0} broadcast(dynamic-slice.48), dimensions={0,1}
  reshape.56 = s32[2]{0} reshape(broadcast.55)
  broadcast.57 = s32[2,4]{1,0} broadcast(reshape.56), dimensions={0}, sharding={devices=[2,1]<=[2]}
  add.58 = s32[2,4]{1,0} add(add.54, broadcast.57), sharding={devices=[2,1]<=[2]}
  reduce.59 = s32[4]{0} reduce(add.58, constant.34), dimensions={0}, to_apply=region_1.19, sharding={replicated}
  add.60 = s32[4]{0} add(get-tuple-element.27, reduce.59), sharding={replicated}
  reshape.60 = s32[4]{0} reshape(add.60), sharding={replicated}
  convert.60 = s32[4]{0} convert(reshape.60), sharding={replicated}
  ROOT tuple.62 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) tuple(add.61, add.50, convert.60, get-tuple-element.28, get-tuple-element.29, /*index=5*/get-tuple-element.30)
} // region_0.23

region_2.63 {
  arg_tuple.64 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) parameter(0)
  get-tuple-element.66 = s32[] get-tuple-element(arg_tuple.64), index=1
  get-tuple-element.67 = s32[4]{0} get-tuple-element(arg_tuple.64), index=2
  get-tuple-element.68 = s32[4,4]{1,0} get-tuple-element(arg_tuple.64), index=3
  get-tuple-element.69 = s32[4,1]{1,0} get-tuple-element(arg_tuple.64), index=4
  get-tuple-element.70 = s32[4,1]{1,0} get-tuple-element(arg_tuple.64), index=5
  get-tuple-element.65 = s32[] get-tuple-element(arg_tuple.64), index=0
  constant.71 = s32[] constant(2)
  ROOT compare.72 = pred[] compare(get-tuple-element.65, constant.71), direction=LT
} // region_2.63

region_3.80 {
  Arg_0.81 = s32[] parameter(0)
  Arg_1.82 = s32[] parameter(1)
  ROOT add.83 = s32[] add(Arg_0.81, Arg_1.82)
}

ENTRY main.87 {
  constant.10 = s32[] constant(0)
  constant.4 = s32[] constant(0)
  broadcast.5 = s32[4]{0} broadcast(constant.4), dimensions={}
  Arg_0.1 = s32[4,4]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  constant.8 = s32[] constant(2)
  broadcast.9 = s32[4,4]{1,0} broadcast(constant.8), dimensions={}
  multiply.11 = s32[4,4]{1,0} multiply(Arg_0.1, broadcast.9), sharding={devices=[2,1]<=[2]}
  Arg_1.2 = s32[4,1]{1,0} parameter(1), sharding={devices=[2,1]<=[2]}
  constant.6 = s32[] constant(2)
  broadcast.7 = s32[4,1]{1,0} broadcast(constant.6), dimensions={}
  multiply.12 = s32[4,1]{1,0} multiply(Arg_1.2, broadcast.7)
  Arg_2.3 = s32[4,1]{1,0} parameter(2), sharding={devices=[2,1]<=[2]}
  multiply.13 = s32[4,1]{1,0} multiply(Arg_2.3, broadcast.7), sharding={devices=[2,1]<=[2]}
  tuple.18 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) tuple(constant.10, constant.10, broadcast.5, multiply.11, multiply.12, multiply.13)
  while.73 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) while(tuple.18), condition=region_2.63, body=region_0.23
  get-tuple-element.74 = s32[] get-tuple-element(while.73), index=0
  get-tuple-element.75 = s32[] get-tuple-element(while.73), index=1
  get-tuple-element.77 = s32[4,4]{1,0} get-tuple-element(while.73), index=3, sharding={devices=[2,1]<=[2]}
  get-tuple-element.78 = s32[4,1]{1,0} get-tuple-element(while.73), index=4, sharding={devices=[2,1]<=[2]}
  get-tuple-element.79 = s32[4,1]{1,0} get-tuple-element(while.73), index=5, sharding={devices=[2,1]<=[2]}
  get-tuple-element.76 = s32[4]{0} get-tuple-element(while.73), index=2
  reduce.84 = s32[] reduce(get-tuple-element.76, constant.10), dimensions={0}, to_apply=region_3.80
  broadcast.84 = s32[4,4] broadcast(reduce.84), dimensions={}
  add.0 = s32[4,4] add(Arg_0.1, broadcast.84), sharding={devices=[2,1]<=[2]}
  add.1 = s32[4,4] add(Arg_1.2, broadcast.84), sharding={devices=[2,1]<=[2]}
  add.2 = s32[4,4] add(Arg_2.3, broadcast.84), sharding={devices=[2,1]<=[2]}
  ROOT get-tuple-element.86 = (s32[4,4], s32[4,4], s32[4,4]) tuple(add.0, add.1, add.2)
} // main.87
)";

TEST_F(MpmdShardMapReduceTest, SimpleShardedContraction) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kSimpleShardedConstractionHlo, /*num_devices=*/2));

  MpmdShardMapLoopReduce mapper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, mapper.Run(module.get()));

  EXPECT_THAT(m::FlatInstructions(module.get()),
              AllOf(Contains(op::CustomCall("SPMDFullToShardShape")).Times(2),
                    Contains(op::CustomCall("SPMDShardToFullShape")).Times(1)));
}

TEST_F(MpmdShardMapReduceTest, DataParallelTransformer) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("data_parallel_transformer.txt", /*num_devices=*/8));

  MpmdShardMapLoopReduce mapper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, mapper.Run(module.get()));

  // TODO: add expects
}

TEST_F(MpmdShardMapReduceTest, FailedShardMapReduce) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("failed_shard_map_reduce.txt", /*num_devices=*/8));

  MpmdShardMapLoopReduce mapper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, mapper.Run(module.get()));

  // TODO: add expects
}

TEST_F(MpmdShardMapReduceTest, ReduceNotShardMapped) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("reduce_not_shard_mapped.txt", /*num_devices=*/8));

  MpmdShardMapLoopReduce mapper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, mapper.Run(module.get()));

  // TODO: add expects, all the reduces of size 51200
  // that are sharded over cxn dims should be replaced
}

constexpr absl::string_view kChannelIdHlo = R"(
region_1.19 {
  Arg_0.20 = s32[] parameter(0)
  Arg_1.21 = s32[] parameter(1)
  ROOT add.22 = s32[] add(Arg_0.20, Arg_1.21)
}

region_0.23 {
  arg_tuple.24 = (s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  get-tuple-element.25 = s32[] get-tuple-element(arg_tuple.24), index=0
  get-tuple-element.26 = s32[] get-tuple-element(arg_tuple.24), index=1
  constant.32 = s32[] constant(0)
  get-tuple-element.27 = s32[4,4]{1,0} get-tuple-element(arg_tuple.24), index=2, sharding={devices=[2,1]<=[2]}
  get-tuple-element.28 = s32[4,4]{1,0} get-tuple-element(arg_tuple.24), index=3, sharding={devices=[2,1]<=[2]}
  reduce.27 = s32[] reduce(get-tuple-element.27, constant.32), dimensions={0,1}, to_apply=region_1.19
  reduce.28 = s32[] reduce(get-tuple-element.28, constant.32), dimensions={0,1}, to_apply=region_1.19
  add.27 = s32[] add(reduce.27, get-tuple-element.25)
  add.28 = s32[] add(reduce.28, get-tuple-element.26)
  ROOT tuple.62 = (s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(add.27, add.28, get-tuple-element.27, get-tuple-element.28)
} // region_0.23

region_2.63 {
  arg_tuple.64 = (s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}) parameter(0)
  ROOT constant.71 = pred[] constant(true)
} // region_2.63

ENTRY main.87 {
  constant.10 = s32[] constant(0)
  constant.4 = s32[] constant(0)
  broadcast.5 = s32[4]{0} broadcast(constant.4), dimensions={}
  Arg_0.1 = s32[4,4]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  constant.8 = s32[] constant(2)
  broadcast.9 = s32[4,4]{1,0} broadcast(constant.8), dimensions={}
  multiply.11 = s32[4,4]{1,0} multiply(Arg_0.1, broadcast.9), sharding={devices=[2,1]<=[2]}
  Arg_1.2 = s32[4,1]{1,0} parameter(1), sharding={devices=[2,1]<=[2]}
  constant.6 = s32[] constant(2)
  broadcast.7 = s32[4,1]{1,0} broadcast(constant.6), dimensions={}
  multiply.12 = s32[4,1]{1,0} multiply(Arg_1.2, broadcast.7)
  Arg_2.3 = s32[4,1]{1,0} parameter(2), sharding={devices=[2,1]<=[2]}
  multiply.13 = s32[4,1]{1,0} multiply(Arg_2.3, broadcast.7), sharding={devices=[2,1]<=[2]}
  tuple.18 = (s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}) tuple(constant.10, constant.10, Arg_0.1, Arg_1.2)
while.73 = (s32[], s32[], s32[4,4]{1,0}, s32[4,4]{1,0}) while(tuple.18), condition=region_2.63, body=region_0.23
  get-tuple-element.74 = s32[] get-tuple-element(while.73), index=0
  get-tuple-element.75 = s32[] get-tuple-element(while.73), index=1
  ROOT get-tuple-element.86 = (s32[], s32[]) tuple(get-tuple-element.74, get-tuple-element.75)
} // main.87
)";

TEST_F(MpmdShardMapReduceTest, UniqueChannelIDs) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kChannelIdHlo, /*num_devices=*/8));

  MpmdShardMapLoopReduce mapper{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, mapper.Run(module.get()));

  // TODO: add expects, all the reduces should have unique channel IDs
}

}  // namespace
}  // namespace xla
