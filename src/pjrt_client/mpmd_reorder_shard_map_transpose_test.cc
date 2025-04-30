/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_reorder_shard_map_transpose.h"
#include "xla/pjrt/legate/mpmd_hoist_shard_map_reduce.h"

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdReorderShardMapTransposeTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

TEST_F(MpmdReorderShardMapTransposeTest, ReorderOnDataParallelTransformer) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("shard-mapped-allreduce-transformer.txt",
                           /*num_devices=*/8));

  // just to get the intermediate hlo for the pass we are interested in
  MpmdHoistShardMapReduce hoister{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));

  EXPECT_THAT(m::FlatInstructions(module.get()),
              Contains(op::Transpose(op::CustomCall("SPMDShardToFullShape"))));

  // perform reordering
  MpmdReorderShardMapTranspose reorderer;
  TF_ASSERT_OK_AND_ASSIGN(changed, reorderer.Run(module.get()));

  EXPECT_THAT(
      m::FlatInstructions(module.get()),
      Contains(op::CustomCall("SPMDShardToFullShape", op::Transpose())));
}

}  // namespace
}  // namespace xla
