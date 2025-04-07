/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_unused_loop_output_remover.h"

#include "gmock/gmock.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdUnusedLoopOutputRemoverTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

using ::testing::Ge;
using ::testing::Lt;

TEST_F(MpmdUnusedLoopOutputRemoverTest, TransformerUnusedOutputs) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("unused_loop_outputs.txt", /*num_devices=*/2));

  MpmdUnusedLoopOutputRemover remover{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, remover.Run(module.get()));

  // TODO: add expects
}

}  // namespace
}  // namespace xla
