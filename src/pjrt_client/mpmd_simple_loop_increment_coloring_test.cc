/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mpmd_simple_loop_increment_coloring.h"

#include "gmock/gmock.h"
#include "xla/pjrt/multimesh/mpmd_microbatch_loop_canonicalizer.h"
#include "xla/pjrt/multimesh/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdSimpleLoopIncrementColoringTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

TEST_F(MpmdSimpleLoopIncrementColoringTest, BasicRecomputeFromArguments) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("2nodes_transformer.txt", /*num_devices=*/2));

  MpmdMicrobatchLoopCanonicalizer inliner{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, inliner.Run(module.get()));

  MpmdSimpleLoopIncrementColoring coloring{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(changed, coloring.Run(module.get()));
}

}  // namespace
}  // namespace xla
