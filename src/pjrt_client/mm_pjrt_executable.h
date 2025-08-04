/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MM_EXECUTABLE_H_
#define XLA_PJRT_MULTIMESH_MM_EXECUTABLE_H_

#include <memory>
#include <vector>

#include "xla/pjrt/multimesh/mm_pjrt_client.h"
#include "xla/pjrt/multimesh/store_handle_fwd.h"
#include "xla/pjrt/multimesh/zuku_execute_context.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/computation_placer.h"

namespace xla {

class WrapperPjRtExecutable : public PjRtLoadedExecutable {
 public:
  WrapperPjRtExecutable(std::unique_ptr<PjRtLoadedExecutable> wrapped,
                        MultiMeshClient* mm_client, ProgramShape program_shape)
      : wrapped_(std::move(wrapped)),
        mm_client_(mm_client),
        program_shape_(std::move(program_shape)) {}

  ~WrapperPjRtExecutable() override = default;

  absl::string_view name() const override { return wrapped_->name(); }

  PjRtClient* client() const override { return wrapped_->client(); }

  int num_replicas() const override { return wrapped_->num_replicas(); }

  int num_partitions() const override { return wrapped_->num_partitions(); }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return wrapped_->FingerprintExecutable();
  }

  int64_t SizeOfGeneratedCodeInBytes() const override {
    return wrapped_->SizeOfGeneratedCodeInBytes();
  }

  const DeviceAssignment& device_assignment() const override {
    return wrapped_->device_assignment();
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    return wrapped_->addressable_device_logical_ids();
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return wrapped_->addressable_devices();
  }

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<>>>& returned_futures) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetParameterLayouts() const override {
    return wrapped_->GetParameterLayouts();
  }

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetOutputLayouts() const override {
    return wrapped_->GetOutputLayouts();
  }

  std::optional<std::vector<OpSharding>> GetParameterShardings()
      const override {
    return wrapped_->GetParameterShardings();
  }

  absl::StatusOr<std::vector<Shape>> GetOutputShapes() const override {
    return wrapped_->GetOutputShapes();
  }

  std::optional<std::vector<OpSharding>> GetOutputShardings() const override {
    return wrapped_->GetOutputShardings();
  }

  void Delete() override { return wrapped_->Delete(); }

  bool IsDeleted() override { return wrapped_->IsDeleted(); }

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    return wrapped_->GetOutputMemoryKinds();
  }

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return wrapped_->GetHloModules();
  }

 private:
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> Wrap(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> outputs);

  std::vector<std::unique_ptr<PjRtBuffer>> Wrap(
      std::vector<std::unique_ptr<PjRtBuffer>> outputs);

  std::unique_ptr<PjRtLoadedExecutable> wrapped_;

  MultiMeshClient* mm_client_;

  ProgramShape program_shape_;
};

class MultiMeshPjRtExecutable : public PjRtLoadedExecutable {
 public:
  MultiMeshPjRtExecutable(
      MultiMeshClient* mm_client, PjRtClient* base_client,
      absl::string_view name, ProgramShape program_shape,
      std::vector<MpmdOperation> schedule, std::vector<Store> temporaries,
      std::shared_ptr<DeviceAssignment> device_assignment,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices, const Shape& result_shape,
      const std::vector<Layout>& parameter_layouts,
      const std::vector<Layout>& output_layouts,
      std::optional<std::vector<OpSharding>> parameter_shardings,
      const std::vector<Shape>& output_shapes,
      std::optional<std::vector<OpSharding>> output_shardings,
      std::string fingerprint,
      std::unique_ptr<WrapperPjRtExecutable> fast_path_exe = nullptr);

  ~MultiMeshPjRtExecutable() override = default;

  absl::string_view name() const override { return name_; }

  PjRtClient* client() const override;

  MultiMeshClient* mm_client() const { return mm_client_; }

  PjRtClient* base_client() const { return base_client_; }

  int num_replicas() const override;

  int num_partitions() const override;

  int64_t SizeOfGeneratedCodeInBytes() const override;

  const ProgramShape& program_shape() const { return program_shape_; }

  const DeviceAssignment& device_assignment() const override;

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override;

  absl::Span<PjRtDevice* const> addressable_devices() const override;

  absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      std::optional<std::vector<PjRtFuture<>>>& returned_futures) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      std::optional<PjRtFuture<>>& returned_future, bool fill_future) override;

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetParameterLayouts() const override;

  absl::StatusOr<std::vector<std::shared_ptr<const PjRtLayout>>>
  GetOutputLayouts() const override;

  std::optional<std::vector<OpSharding>> GetParameterShardings()
      const override {
    return parameter_shardings_;
  }

  absl::StatusOr<std::vector<Shape>> GetOutputShapes() const override {
    return output_shapes_;
  }

  std::optional<std::vector<OpSharding>> GetOutputShardings() const override {
    return output_shardings_;
  }

  absl::StatusOr<std::string> FingerprintExecutable() const override {
    return fingerprint_;
  }

  void Delete() override;

  bool IsDeleted() override;

  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  const std::vector<MpmdOperation>& schedule() const { return schedule_; }

 private:
  std::vector<MpmdOperation> schedule_;

  std::string name_;

  PjRtClient* base_client_;

  MultiMeshClient* mm_client_;

  ProgramShape program_shape_;

  Shape result_shape_;

  std::shared_ptr<DeviceAssignment> device_assignment_;
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;
  std::vector<PjRtDevice*> addressable_devices_;

  std::unique_ptr<WrapperPjRtExecutable> fast_path_exe_;

  std::vector<Layout> parameter_layouts_;
  std::vector<Layout> output_layouts_;
  std::vector<Shape> output_shapes_;
  std::optional<std::vector<OpSharding>> parameter_shardings_;
  std::optional<std::vector<OpSharding>> output_shardings_;
  std::vector<Store> temporaries_;
  std::vector<std::vector<StoreHandle>> cached_temporary_stores_;
  std::vector<zuku::Future<zuku::ArrayTile>> temp_allocations_;
  std::shared_ptr<ZukuExecuteContext> context_;
  std::string fingerprint_;

  bool deleted_ = false;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_MM_EXECUTABLE_H_
