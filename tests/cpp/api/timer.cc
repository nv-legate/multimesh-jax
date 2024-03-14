/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "legate_xla_utils.h"
#include "gmock/gmock.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <legate_xla_common.h>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <xla_to_legate.h>

namespace legate_xla {
namespace {

using ::testing::ElementsAre;

class TestExecutable : public LegateExecutable {
public:
  std::optional<std::string>
  Execute(uint64_t run_id, const std::vector<BufferAllocation> &inputs,
          const std::vector<BufferAllocation> &outputs,
          TaskMemoryAllocator *allocator,
          const DeviceAssignment &device_assignment, bool cpu) const override {
    sleep(2);
    return std::nullopt;
  }

  std::pair<int, int> MachineSlice() const override { return {0, 1}; }

  size_t LaunchSize() const override { return 1; }

  int ReplicaCount() const override { return 1; }

  int NumPartitions() const override { return 1; }

  std::string Name() const override { return "test-executable"; }
};

class TestCompiler : public LegateCompiler {
public:
  void Compile(uint64_t run_id, const CompileConfig &config) override {}

  std::unique_ptr<LegateExecutable> MakeExecutable() override {
    return std::unique_ptr<LegateExecutable>(new TestExecutable);
  }

  std::pair<int, int> MachineSlice() const override { return {0, 1}; }

  size_t LaunchSize() const override { return 1; }

  std::string Name() const override { return "TestCompiler"; }

  uint64_t HloId() const override { return 0; }
};

MATCHER_P(ScalarBufferIs, scalar, "") {
  using Scalar = std::decay_t<decltype(scalar)>;
  Scalar value;
  cudaMemcpy(&value, arg.buffer, sizeof(Scalar), cudaMemcpyDeviceToHost);
  return value == scalar;
}

TEST(TimerTest, BasicTimer) {
  std::shared_ptr<LegateCompiler> compiler = std::make_shared<TestCompiler>();
  auto *hold = Hold(compiler);

  // just validate that this runs without trouble

  legate_xla::StartTimer("first");
  legate_xla::CreateExecuteTask(hold, {}, {}, {}, nullptr);
  legate_xla::StopTimer("first");
  legate_xla::StartTimer("second");
  legate_xla::CreateExecuteTask(hold, {}, {}, {}, nullptr);
  legate_xla::StopTimer("second");
}

} // namespace
} // namespace legate_xla