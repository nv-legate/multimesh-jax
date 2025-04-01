/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_LEGATE_BUFFER_H_
#define XLA_PJRT_LEGATE_BUFFER_H_

#include <variant>

#include "xla/pjrt/legate/legate_buffer_action.h"
#include "xla/pjrt/legate/legate_computation.h"
#include "xla/pjrt/legate/legate_pjrt_client.h"
#include "xla/util.h"

namespace xla {

/*
 * Notes:
 *   - provides Legate store for task creation
 *   - implements PjRt interface
 *   - no shape tuples supported / deactivated upon HLO-compile
 *
 * Lifecycle
 *   - created from output / store --> delete() provided for PjRtBuffer but NOOP
 *
 */

class LegatePjRtBuffer : public PjRtBuffer {
 public:
  using data_variant_t = std::variant<StoreHandle, BufferFromHostBufferAction*,
                                      std::shared_ptr<PjRtBuffer>>;
  LegatePjRtBuffer(data_variant_t data, std::optional<HloSharding> sharding,
                   Shape global_shape, Shape local_shape,
                   LegateClient* legate_client, PjRtClient* base_client,
                   PjRtDevice* device, PjRtMemorySpace* memory_space,
                   std::optional<std::string> name = std::nullopt)
      : data_(std::move(data)),
        sharding_(std::move(sharding)),
        global_shape_(std::move(global_shape)),
        local_shape_(std::move(local_shape)),
        legate_client_(legate_client),
        base_client_(base_client),
        memory_space_(memory_space),
        device_(device),
        name_(std::move(name)) {}

  ~LegatePjRtBuffer() override;

  LegatePjRtBuffer(const LegatePjRtBuffer&) = delete;
  LegatePjRtBuffer(LegatePjRtBuffer&&) = delete;
  LegatePjRtBuffer& operator=(const LegatePjRtBuffer&) = delete;
  LegatePjRtBuffer& operator=(LegatePjRtBuffer&&) = delete;

  PjRtFuture<> LazyToLiteral(
      absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator)
      override;

  bool has_store() const { return std::holds_alternative<StoreHandle>(data_); }

  const StoreHandle& store() const {
    if (deleted_) LOG(ERROR) << "ERROR: Accessing store of deleted Buffer!";
    if (!has_store()) {
      LOG(FATAL) << "Accessing LegatePjRtBuffer that is not backed by a store";
    }
    return std::get<StoreHandle>(data_);
  }

  absl::StatusOr<StoreHandle> ToStore(
      const zuku::ShardedShape& legate_shape,
      const std::shared_ptr<LegateStream>& stream,
      std::optional<StoreHandle> existing_store = std::nullopt) const;

  const Shape& on_device_shape() const override { return local_shape_; }

  const Shape& global_shape() const { return global_shape_; }

  // FIXME dynamic shapes?
  absl::StatusOr<Shape> logical_on_device_shape() override {
    return local_shape_;
  }

  PjRtDevice* device() const override { return device_; }

  PjRtClient* client() const override { return legate_client_; }

  PjRtMemorySpace* memory_space() const override { return memory_space_; }

  void set_store(const StoreHandle& store) { data_ = store; }

  std::string name() const override {
    return name_.has_value() ? *name_ : "unnamed";
  }

  absl::Status ResolveHostAction();

  bool has_host_action() const {
    return std::holds_alternative<BufferFromHostBufferAction*>(data_);
  }

  BufferFromHostBufferAction* host_action() const {
    return std::get<BufferFromHostBufferAction*>(data_);
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
    LOG(ERROR) << "CopyToRemoveDevice not supported in LegateBuffer";
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
  LegateClient* legate_client_;
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

#endif  // XLA_PJRT_LEGATE_BUFFER_H_
