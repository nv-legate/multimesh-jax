/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_TEST_BASE_H_
#define XLA_PJRT_LEGATE_TEST_BASE_H_

#include <filesystem>

#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/legate/legate_mock_legate_xla.h"
#include "xla/pjrt/legate/legate_pjrt_client.h"
#include "xla/pjrt/legate/legate_pjrt_executable.h"
#include "xla/service/computation_placer.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {

class LegateTestBase : public HloTestBase {
 public:
  struct Config {
    bool use_module_config_auto_output_sharding{false};
    bool use_auto_input_sharding{false};
    bool hoist_loop_convert{false};
    bool remove_hoisted_reduces{false};
  };

  LegateTestBase();

  absl::StatusOr<HloSharding> GetSharding(absl::string_view pbtxt);

  static void SetUpTestSuite();

  static void TearDownTestSuite();

  void SetUp() override;

  void TearDown() override;

  DeviceAssignment GetDeviceAssignment(int num_devices);

  bool Skip(int num_devices) const;

  absl::StatusOr<std::unique_ptr<LegatePjRtExecutable>> Compile(
      absl::string_view hlo_string, int num_devices, Config cfg);

 protected:
  void LegateMockReset();

  int64_t DeviceBytesHighWatermark(int64_t local_device_id) const;

  int64_t HostBytesHighWatermark(int64_t local_device_id) const;

  void SetCompileModules(bool flag);

  int64_t LegateMockStateHash() const;

  void LegateMockSetComputeHashes(bool flag);

  MockZukuExecuteContext* context() const;

  absl::StatusOr<std::string> GetFileText(absl::string_view relative_path);

  static se::Platform* platform_;
  static std::unique_ptr<Backend> backend_;
  static std::unique_ptr<LegateClient> client_;
  std::filesystem::path testdata_root_;
};

}  // namespace xla

#endif
