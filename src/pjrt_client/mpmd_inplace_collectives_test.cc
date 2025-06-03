/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_inplace_collectives.h"

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdInplaceCollectivesTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

using ::testing::Gt;

TEST_F(MpmdInplaceCollectivesTest, SimpleImplicitTask) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("gpt3_inplace_collectives.txt", /*num_devices=*/32));

  MpmdInPlaceCollectives placer{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, placer.Run(module.get()));
  EXPECT_TRUE(changed);
}

}  // namespace
}  // namespace xla
