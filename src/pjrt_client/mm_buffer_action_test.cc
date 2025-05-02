/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mm_buffer_action.h"

#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/multimesh/cuda_utils.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/tsl/platform/test.h"

static constexpr int32_t kBasicTestNumElements = 4;

namespace xla {
namespace {

class BufferActionTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() {
    TF_ASSERT_OK_AND_ASSIGN(
        auto gpu_client,
        GetStreamExecutorGpuClient(
            {.allocator_config = {.memory_fraction = 0.50, .preallocate = true},
             .node_id = 0,
             .num_nodes = 1}));
    auto* se_client =
        dynamic_cast<PjRtStreamExecutorClient*>(gpu_client.release());
    ASSERT_FALSE(se_client == nullptr);

    client_ = std::unique_ptr<PjRtStreamExecutorClient>(se_client);

    cuda_utils::Init();
    for (int i = 0; i < client_->addressable_device_count(); ++i) {
      cuda_utils::SetPrimaryContext(i);
    }
  }

  static void TearDownTestSuite() {
    // make sure this gets torn down before program end
    // otherwise the cuda driver will no longer be valid
    client_ = nullptr;
  }

 protected:
  static std::unique_ptr<PjRtStreamExecutorClient> client_;
};

std::unique_ptr<PjRtStreamExecutorClient> BufferActionTest::client_;

TEST_F(BufferActionTest, BasicTest) {
  std::vector<int32_t> host_data(kBasicTestNumElements);
  std::iota(host_data.begin(), host_data.end(), 0);

  std::vector<int64_t> dims = {kBasicTestNumElements, 1};

  BufferFromHostBufferAction action{
      host_data.data(),
      PrimitiveType::S32,
      dims,
      /*byte_strides=*/std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
      nullptr,
      client_.get()};

  void* device_buffer =
      cuda_utils::AllocateDeviceMemory(kBasicTestNumElements * sizeof(int32_t));
  action.Act(device_buffer, 0);

  std::vector<int32_t> final_host_data(kBasicTestNumElements);
  cuda_utils::CopyDeviceToHost(final_host_data.data(), device_buffer,
                               kBasicTestNumElements * sizeof(int32_t));

  EXPECT_EQ(host_data, final_host_data);
}

TEST_F(BufferActionTest, OnDoneCallback) {
  std::vector<int32_t> host_data(kBasicTestNumElements);
  std::iota(host_data.begin(), host_data.end(), 0);

  std::vector<int64_t> dims = {kBasicTestNumElements, 1};

  bool callback_done = false;
  BufferFromHostBufferAction action{
      host_data.data(),
      PrimitiveType::S32,
      dims,
      /*byte_strides=*/std::nullopt,
      PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall,
      [&] { callback_done = true; },
      client_.get()};

  void* device_buffer =
      cuda_utils::AllocateDeviceMemory(kBasicTestNumElements * sizeof(int32_t));
  action.Act(device_buffer, 0);

  EXPECT_TRUE(callback_done);
}

}  // namespace
}  // namespace xla
