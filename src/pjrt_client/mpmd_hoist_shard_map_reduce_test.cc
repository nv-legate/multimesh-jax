/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_hoist_shard_map_reduce.h"

#include "gmock/gmock.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdHoistShardMapReduceTest : public MpmdTestBase {};

TEST_F(MpmdHoistShardMapReduceTest, DataParallelTransformer) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("shard-mapped-allreduce-transformer.txt",
                           /*num_devices=*/8));

  MpmdHoistShardMapReduce hoister{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));
}

}  // namespace
}  // namespace xla
