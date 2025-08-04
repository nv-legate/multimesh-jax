/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_BUFFER_H_
#define XLA_PJRT_MULTIMESH_BUFFER_H_

#include <variant>

#include "xla/pjrt/multimesh/mm_computation.h"
#include "xla/pjrt/multimesh/mm_pjrt_client.h"
#include "xla/util.h"

namespace xla {

/*
 * Notes:
 *   - provides store for task creation
 *   - implements PjRt interface
 *   - no shape tuples supported / deactivated upon HLO-compile
 *
 * Lifecycle
 *   - created from output / store --> delete() provided for PjRtBuffer but NOOP
 *
 */

class MultiMeshPjRtBuffer : public PjRtBuffer {
 public:
  using data_variant_t = std::variant<StoreHandle, std::shared_ptr<PjRtBuffer>>;
  MultiMeshPjRtBuffer(data_variant_t data, std::optional<HloSharding> sharding,
                      Shape global_shape, Shape local_shape,
                      MultiMeshClient* mm_client, PjRtClient* base_client,
                      PjRtDevice* device, PjRtMemorySpace* memory_space,
                      std::optional<std::string> name = std::nullopt)
      : data_(std::move(data)),
        sharding_(std::move(sharding)),
        global_shape_(std::move(global_shape)),
        local_shape_(std::move(local_shape)),
        mm_client_(mm_client),
        base_client_(base_client),
        memory_space_(memory_space),
        device_(device),
        name_(std::move(name)) {}

  ~MultiMeshPjRtBuffer() override;

  MultiMeshPjRtBuffer(const MultiMeshPjRtBuffer&) = delete;
  MultiMeshPjRtBuffer(MultiMeshPjRtBuffer&&) = delete;
  MultiMeshPjRtBuffer& operator=(const MultiMeshPjRtBuffer&) = delete;
  MultiMeshPjRtBuffer& operator=(MultiMeshPjRtBuffer&&) = delete;

  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override;

  bool has_store() const { return std::holds_alternative<StoreHandle>(data_); }

  const StoreHandle& store() const {
    if (deleted_) LOG(ERROR) << "ERROR: Accessing store of deleted Buffer!";
    if (!has_store()) {
      LOG(FATAL)
          << "Accessing MultiMeshPjRtBuffer that is not backed by a store";
    }
    return std::get<StoreHandle>(data_);
  }

  absl::StatusOr<StoreHandle> ToStore(
      const zuku::ShardedShape& mm_shape,
      const std::shared_ptr<MultiMeshStream>& stream,
      std::optional<StoreHandle> existing_store = std::nullopt) const;

  const Shape& on_device_shape() const override { return local_shape_; }

  const Shape& global_shape() const { return global_shape_; }

  // FIXME dynamic shapes?
  absl::StatusOr<Shape> logical_on_device_shape() override {
    return local_shape_;
  }

  PjRtDevice* device() const override { return device_; }

  PjRtClient* client() const override { return mm_client_; }

  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  void set_store(const StoreHandle& store) { data_ = store; }

  std::string name() const override {
    return name_.has_value() ? *name_ : "unnamed";
  }

  bool has_native_buffer() const {
    return std::holds_alternative<std::shared_ptr<PjRtBuffer>>(data_);
  }

  PjRtBuffer* native_buffer() const {
    return std::get<std::shared_ptr<PjRtBuffer>>(data_).get();
  }

  std::shared_ptr<PjRtBuffer> shared_native_buffer() const {
    return std::get<std::shared_ptr<PjRtBuffer>>(data_);
  }

  absl::StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  bool HasValidShard() const override;

  PjRtFuture<> ToLiteral(MutableLiteralBase* literal) override;

  absl::StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  PjRtFuture<> CopyRawToHost(void* dst, int64_t offset,
                             int64_t transfer_size) override {
    return PjRtFuture<>(Unimplemented("CopyRawToHost"));
  }

  void Delete() override { deleted_ = true; }

  absl::StatusOr<std::unique_ptr<ExternalReference>>
  ReleaseDeviceMemoryOwnership(bool wait_for_operations_to_complete) override {
    return Unimplemented("ReleaseDeviceMemoryOwnership");
  }

  bool IsDeleted() override { return deleted_; }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> CopyToMemorySpace(
      PjRtMemorySpace* dst_memory_space) override;

  void CopyToRemoteDevice(PjRtFuture<std::string> serialized_descriptor,
                          RemoteSendCallback on_done) override {
    LOG(ERROR) << "CopyToRemoveDevice not supported in MultiMesh";
  }

  PjRtFuture<> GetReadyFuture() override;

  bool IsOnCpu() const override {
    auto res = base_client_->platform_id() == CpuId();
    return res;
  }

  bool has_sharding() const { return sharding_.has_value(); }

  const HloSharding& sharding() const { return *sharding_; }

  void attach(std::shared_ptr<PjRtBuffer> to_attach) {
    attached_ = std::move(to_attach);
  }

 private:
  absl::StatusOr<std::unique_ptr<PjRtBuffer>> SliceLocalShard() const;

  std::pair<int64_t, int64_t> ShardingDeviceRange() const;

  std::pair<int64_t, int64_t> SliceDeviceRange() const;

  bool Distributed() const {
    return base_client_->device_count() !=
           base_client_->addressable_device_count();
  }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> SliceStore() const;

  data_variant_t data_;
  Shape global_shape_;
  Shape local_shape_;
  MultiMeshClient* mm_client_;
  PjRtClient* base_client_;
  PjRtDevice* device_;
  PjRtMemorySpace* memory_space_;
  std::shared_ptr<PjRtBuffer> attached_;
  std::optional<std::string> name_;
  std::optional<HloSharding> sharding_;

  bool deleted_ = false;
  bool donated_ = false;
};

}  // namespace xla

#endif  // XLA_PJRT_MULTIMESH_BUFFER_H_
