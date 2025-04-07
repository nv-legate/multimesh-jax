/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef XLA_PJRT_LEGATE_CLIENT_H_
#define XLA_PJRT_LEGATE_CLIENT_H_

#include "xla/layout.h"
#include "xla/pjrt/legate/mpmd_partition.h"
#include "xla/pjrt/legate/zuku_execute_context.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/pjrt/transpose.h"

namespace xla {

class LegateClient : public PjRtClient {
 public:
  explicit LegateClient(std::unique_ptr<PjRtClient> base_client,
                        xla::Backend* backend,
                        std::shared_ptr<ZukuExecuteContext> context);

  ~LegateClient() override;

 public:
  ZukuExecuteContext* mutable_context() const { return context_.get(); }

  const std::shared_ptr<ZukuExecuteContext>& shared_context() const {
    return context_;
  }

  xla::Backend* backend() const { return backend_; }

  int process_index() const override { return base_client_->process_index(); }

  int device_count() const override { return base_client_->device_count(); }

  int addressable_device_count() const override {
    return base_client_->addressable_device_count();
  }

  absl::Span<PjRtDevice* const> devices() const override {
    return base_client_->devices();
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return base_client_->addressable_devices();
  }

  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return base_client_->memory_spaces();
  }

  absl::StatusOr<PjRtDevice*> LookupDevice(
      PjRtGlobalDeviceId device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      PjRtLocalDeviceId local_hardware_id) const override;

  PjRtPlatformId platform_id() const override { return platform_id_; }

  absl::string_view platform_name() const override { return platform_name_; }

  absl::string_view platform_version() const override;

  HloModuleProto ShardBatch(const HloModuleProto& proto) const;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis()
      const override {
    return base_client_->GetHloCostAnalysis();
  }

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  // Extra entry point for testing that avoids querying global devices
  // to compute the max addressable devices per process
  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options,
      std::optional<int64_t> max_per_process, bool only_compile_device0,
      MpmdPartitionConfig config = {});

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

  absl::StatusOr<Layout> GetDefaultLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dims) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(
      absl::Span<const ShapeSpec> shapes,
      std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
      PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
  MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) override;

  absl::Status Defragment() override;

 private:
  std::unique_ptr<PjRtClient> base_client_;

  std::shared_ptr<ZukuExecuteContext> context_;

  xla::Backend* backend_;

  // TransposePlanCache to transform data upon buffer creation
  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);

  //  Spoofing 'gpu' here keeps us from modifying capabilities within jax
  //  External code might also use this, e.g. "t5x/t5x/partitioning.py"
  std::string platform_name_{"legate"};

  const PjRtPlatformId platform_id_ = tsl::Fingerprint64(platform_name_);
};

}  // namespace xla

extern "C" void ReplicateParametersSmallerThanNumElements(int64_t num_elements);

extern "C" void RecomputeArgumentsIfCostLessThan(int64_t cost);

#endif  // XLA_PJRT_LEGATE_CLIENT_H_
