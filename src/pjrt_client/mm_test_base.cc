/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mm_test_base.h"

#include <filesystem>
#include <fstream>
#include <utility>

#include "xla/client/executable_build_options.h"
#include "xla/hlo/builder//xla_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mm_mock_mm_xla.h"
#include "xla/pjrt/multimesh/mm_pjrt_executable.h"
#include "xla/pjrt/multimesh/mm_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"

namespace xla {

se::Platform* MultiMeshTestBase::platform_{nullptr};
std::unique_ptr<Backend> MultiMeshTestBase::backend_{nullptr};
std::unique_ptr<MultiMeshClient> MultiMeshTestBase::client_{nullptr};
static constexpr int kMaxNumDevices = 128;

absl::StatusOr<HloSharding> MultiMeshTestBase::GetSharding(
    absl::string_view pbtxt) {
  OpSharding sharding_proto;
  if (!tsl::protobuf::TextFormat::ParseFromString(std::string(pbtxt),
                                                  &sharding_proto)) {
    return InvalidArgument("failed to parse sharding proto");
  }
  return HloSharding::FromProto(sharding_proto);
}

MultiMeshTestBase::MultiMeshTestBase()
    : HloTestBase(*PlatformUtil::GetPlatform("CUDA"),
                  *PlatformUtil::GetPlatform("cpu")) {
  std::filesystem::path this_file = __FILE__;
  testdata_root_ = this_file.parent_path() / "testdata";
}

void MultiMeshTestBase::SetUp() { MultiMeshMockReset(); }

void MultiMeshTestBase::SetUpTestSuite() {
  HloTestBase::SetUpTestSuite();

  auto gpu_platform = PlatformUtil::GetPlatform("CUDA");
  const bool gpu_ok = gpu_platform.ok();
  if (gpu_ok) {
    platform_ = *std::move(gpu_platform);
  } else {
    TF_ASSERT_OK_AND_ASSIGN(platform_, PlatformUtil::GetPlatform("cpu"));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      backend_,
      Backend::CreateBackend(BackendOptions().set_platform(platform_)));

  auto native_client = [&] {
    if (gpu_ok) {
      xla::GpuClientOptions options;
      options.allocator_config = {.memory_fraction = 0.02,
                                  .preallocate = false};
      options.node_id = 0;
      options.num_nodes = 1;
      return GetStreamExecutorGpuClient(options);
    }
    return xla::GetTfrtCpuClient(
        {.asynchronous = false, .cpu_device_count = kMaxNumDevices});
  }();

  client_ = std::make_unique<MultiMeshClient>(
      *std::move(native_client), backend_.get(),
      std::make_shared<MockZukuExecuteContext>());
}

void MultiMeshTestBase::TearDownTestSuite() {
  ClearCachedStreams();
  client_ = nullptr;
  platform_ = nullptr;
  backend_ = nullptr;
}

void MultiMeshTestBase::TearDown() {
  MultiMeshMockReset();
  HloTestBase::TearDown();
  ClearMetadataNameTasks();
}

MockZukuExecuteContext* MultiMeshTestBase::context() const {
  return dynamic_cast<MockZukuExecuteContext*>(client_->mutable_context());
}

void MultiMeshTestBase::MultiMeshMockReset() { context()->Reset(); }

int64_t MultiMeshTestBase::DeviceBytesHighWatermark(
    int64_t local_device_id) const {
  return context()->DeviceBytesHighWatermark(local_device_id);
}

int64_t MultiMeshTestBase::HostBytesHighWatermark(
    int64_t local_device_id) const {
  return context()->HostBytesHighWatermark(local_device_id);
}

void MultiMeshTestBase::SetCompileModules(bool flag) {
  context()->SetCompileModules(flag);
}

int64_t MultiMeshTestBase::MultiMeshMockStateHash() const {
  return context()->OperationHash();
}

void MultiMeshTestBase::MultiMeshMockSetComputeHashes(bool flag) {
  context()->SetComputeHashes(flag);
}

DeviceAssignment MultiMeshTestBase::GetDeviceAssignment(int num_devices) {
  DeviceAssignment da{1, num_devices};
  for (int d = 0; d < num_devices; ++d) {
    da(0, d) = d;
  }
  return da;
}

bool MultiMeshTestBase::Skip(int num_devices) const {
  return client_->device_count() < num_devices;
}

absl::StatusOr<std::string> MultiMeshTestBase::GetFileText(
    absl::string_view relative_path) {
  auto full_path = testdata_root_ / std::string(relative_path);
  std::ifstream ifs(full_path);
  if (ifs) {
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
  }
  return InvalidArgumentStrCat("test file ", relative_path, " does not exist");
}

absl::StatusOr<std::unique_ptr<MultiMeshPjRtExecutable>>
MultiMeshTestBase::Compile(absl::string_view hlo_string, int num_devices,
                           Config cfg) {
  TF_ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(hlo_string));
  HloModuleProto proto = module->ToProto();

  if (num_devices > kMaxNumDevices) {
    return InvalidArgumentStrCat("Compiling with too many devices ",
                                 num_devices, ", max is ", kMaxNumDevices);
  }

  XlaComputation comp(proto);

  HloInstruction* root = module->entry_computation()->root_instruction();
  size_t num_outputs =
      root->shape().IsTuple() ? root->shape().tuple_shapes_size() : 1;

  absl::Span<const bool> allow_sharding_propagation_to_outputs =
      module->config().allow_spmd_sharding_propagation_to_output();
  absl::Span<const bool> allow_sharding_propagation_to_params =
      module->config().allow_spmd_sharding_propagation_to_output();
  bool allow[] = {true};
  if (!cfg.use_module_config_auto_output_sharding) {
    // if not specified, set to
    // true for all outputs
    allow_sharding_propagation_to_outputs = allow;
  }

  if (!cfg.use_module_config_auto_param_sharding) {
    // if not specified, set to
    // true for all params
    allow_sharding_propagation_to_params = allow;
  }

  CompileOptions options{
      .executable_build_options =
          ExecutableBuildOptions()
              .set_allow_spmd_sharding_propagation_to_output(
                  allow_sharding_propagation_to_outputs)
              .set_allow_spmd_sharding_propagation_to_parameters(
                  allow_sharding_propagation_to_params)
              .set_use_auto_spmd_partitioning(cfg.use_auto_input_sharding)};

  options.executable_build_options.set_use_auto_spmd_partitioning(num_devices >
                                                                  1);
  options.executable_build_options.set_num_partitions(num_devices);
  options.executable_build_options.set_num_replicas(1);

  // do not do autotuning with the mock backend
  options.executable_build_options.mutable_debug_options()
      ->set_xla_gpu_autotune_level(0);

  options.executable_build_options.set_device_assignment(
      GetDeviceAssignment(num_devices));

  // force allocation of the debug options
  options.executable_build_options.mutable_debug_options();

  TF_ASSIGN_OR_RETURN(
      auto exe,
      client_->Compile(comp, options, /*max_per_process=*/1,
                       /*only_compile_device0=*/true,
                       {.hoist_loop_convert = cfg.hoist_loop_convert,
                        .remove_hoisted_reduces = cfg.remove_hoisted_reduces}));

  auto* mm_exe = dynamic_cast<MultiMeshPjRtExecutable*>(exe.release());
  if (!mm_exe) {
    return InvalidArgumentStrCat(
        "produced executable that is not a MultiMeshPjRtExecutable");
  }
  return std::unique_ptr<MultiMeshPjRtExecutable>(mm_exe);
}
}  // namespace xla
