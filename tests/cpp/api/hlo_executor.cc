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
#include <stdexcept>
#include <xla_to_legate.h>

namespace legate_xla {
namespace {

using ::testing::ElementsAre;

class TestExecutable : public LegateExecutable {
public:
  TestExecutable(
      std::function<void(const std::vector<BufferAllocation> &inputs,
                         const std::vector<BufferAllocation> &outputs)>
          fxn)
      : fxn_(fxn) {}

  std::optional<std::string>
  Execute(uint64_t run_id, const std::vector<BufferAllocation> &inputs,
          const std::vector<BufferAllocation> &outputs,
          TaskMemoryAllocator *allocator,
          const DeviceAssignment &device_assignment, bool cpu) const override {
    fxn_(inputs, outputs);
    return std::nullopt;
  }

  std::pair<int, int> MachineSlice() const override { return {0, 1}; }

  size_t LaunchSize() const override { return 1; }

  int ReplicaCount() const override { return 1; }

  int NumPartitions() const override { return 1; }

  std::string Name() const override { return "test-executable"; }

private:
  std::function<void(const std::vector<BufferAllocation> &inputs,
                     const std::vector<BufferAllocation> &outputs)>
      fxn_;
};

class TestCompiler : public LegateCompiler {
public:
  TestCompiler(std::function<void(const std::vector<BufferAllocation> &inputs,
                                  const std::vector<BufferAllocation> &outputs)>
                   fxn)
      : fxn_(fxn) {}

  void Compile(uint64_t run_id, const CompileConfig &config) override {}

  std::unique_ptr<LegateExecutable> MakeExecutable() override {
    return std::unique_ptr<LegateExecutable>(new TestExecutable(fxn_));
  }

  std::pair<int, int> MachineSlice() const override { return {0, 1}; }

  size_t LaunchSize() const override { return 1; }

  std::string Name() const override { return "TestCompiler"; }

  uint64_t HloId() const override { return 0; }

private:
  std::function<void(const std::vector<BufferAllocation> &inputs,
                     const std::vector<BufferAllocation> &outputs)>
      fxn_;
};

MATCHER_P(ScalarBufferIs, scalar, "") {
  using Scalar = std::decay_t<decltype(scalar)>;
  Scalar value;
  cudaMemcpy(&value, arg.buffer, sizeof(Scalar), cudaMemcpyDeviceToHost);
  return value == scalar;
}

TEST(HLOExecutorTest, ScalarArgumemts) {

  static constexpr int64_t kScalar0 = 12;
  static constexpr int32_t kScalar1 = 16;

  std::condition_variable cv;
  std::mutex m;
  std::unique_lock lk(m);

  bool ready = false;
  std::shared_ptr<LegateCompiler> compiler = std::make_shared<TestCompiler>(
      [&](const std::vector<BufferAllocation> &inputs,
          const std::vector<BufferAllocation> &outputs) {
        EXPECT_THAT(inputs, ElementsAre(ScalarBufferIs(kScalar0),
                                        ScalarBufferIs(kScalar1)));
        ready = true;
        cv.notify_one();
      });

  auto *hold = Hold(compiler);

  legate_xla::CreateExecuteTask(hold, {{kScalar0, 0}, {kScalar1, 1}}, {}, {},
                                nullptr);

  cv.wait(lk, [&] { return ready; });
}

} // namespace
} // namespace legate_xla