/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mm_pjrt_buffer.h"

#include <memory>
#include <string>

#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/multimesh/mm_utils.h"
#include "xla/pjrt/multimesh/zuku_execute_context.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"

namespace xla {
namespace {

class StoreExternalReference : public PjRtBuffer::ExternalReference {
 public:
  explicit StoreExternalReference(void* data, const StoreHandle store)
      : store_(store) {
    data_ptr_ = data;
  }

 private:
  StoreHandle store_;
};

absl::StatusOr<std::pair<const void*, size_t>> GetPjRtBufferData(
    PjRtBuffer* buffer, se::Stream* stream) {
  if (buffer->IsOnCpu()) {
    TF_ASSIGN_OR_RETURN(auto external_ref, buffer->AcquireExternalReference());
    void* src_buf = external_ref->OpaqueDeviceMemoryDataPointer();
    TF_ASSIGN_OR_RETURN(auto size, buffer->GetOnDeviceSizeInBytes());
    return std::make_pair(src_buf, size);
  }
  auto* se_device = dynamic_cast<PjRtStreamExecutorDevice*>(buffer->device());
  if (!se_device) {
    return InvalidArgument(
        "GetStreamExecutorBuffer: device is not a stream executor device");
  }

  auto* se_client_buffer = dynamic_cast<PjRtStreamExecutorBuffer*>(buffer);
  if (!se_client_buffer) {
    return InvalidArgument(
        "GetStreamExecutorBuffer: buffer is not a stream executor buffer");
  }

  auto hold = se_client_buffer->GetBufferWithUsageHold();

  std::unique_ptr<se::Stream> copy_stream;
  if (stream) {
    WaitForBufferDefinitionEventsOnStream(hold.buffer()->definition_events(),
                                          stream);
  } else {
    auto new_stream = se_device->local_device_state()->BorrowStreamFromPool();
    WaitForBufferDefinitionEventsOnStream(hold.buffer()->definition_events(),
                                          new_stream.get());
    // this will cuStreamSynchronize when the new_stream destructor is called
  }

  const void* ptr = hold.buffer()
                        ->AsShapedBuffer(buffer->on_device_shape())
                        .root_buffer()
                        .opaque();
  const size_t size = hold.buffer()
                          ->AsShapedBuffer(buffer->on_device_shape())
                          .root_buffer()
                          .size();
  return std::make_pair(ptr, size);
}

}  // namespace

PjRtFuture<> MultiMeshPjRtBuffer::ToLiteral(MutableLiteralBase* literal) {
  if (has_native_buffer()) {
    return native_buffer()->ToLiteral(literal);
  }

  auto single_buf = SliceLocalShard();
  if (!single_buf.ok()) {
    return PjRtFuture<>(std::move(single_buf).status());
  }

  return (*single_buf)->ToLiteral(literal);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
MultiMeshPjRtBuffer::CopyToMemorySpace(PjRtMemorySpace* dst_memory_space) {
  auto* dst_device = dst_memory_space->devices().front();
  VLOG(3) << "MultiMeshPjRtBuffer::CopyToDevice: "
          << dst_device->global_device_id().value();
  auto new_variant = [&]() -> absl::StatusOr<data_variant_t> {
    if (has_native_buffer()) {
      TF_ASSIGN_OR_RETURN(auto new_buf,
                          native_buffer()->CopyToMemorySpace(dst_memory_space));
      return std::move(new_buf);
    }
    return store();
  }();

  TF_ASSIGN_OR_RETURN(auto data, std::move(new_variant));
  return std::unique_ptr<PjRtBuffer>(std::make_unique<MultiMeshPjRtBuffer>(
      std::move(data), sharding_, global_shape_, local_shape_, mm_client_,
      base_client_, dst_device,
      dst_device->default_memory_space().value_or(nullptr),
      name() + ".copy-to-device"));
}

MultiMeshPjRtBuffer::~MultiMeshPjRtBuffer() {
  if (has_store()) {
    auto& store = std::get<StoreHandle>(data_);
    mm_client_->mutable_context()->Destroy(store);
  }
}

PjRtFuture<> MultiMeshPjRtBuffer::LazyToLiteral(
    absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator) {
  auto buffer = std::move(generator)();
  if (!buffer.ok()) {
    return PjRtFuture<>(buffer.status());
  }
  return ToLiteral(buffer.value());
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
MultiMeshPjRtBuffer::AcquireExternalReference() {
  if (has_native_buffer()) {
    return native_buffer()->AcquireExternalReference();
  }

  void* data = mm_client_->mutable_context()->SliceLocalShard(
      device_->local_device_id().value(), store());

  return std::unique_ptr<ExternalReference>(
      std::make_unique<StoreExternalReference>(data, store()));
}

absl::StatusOr<StoreHandle> MultiMeshPjRtBuffer::ToStore(
    const zuku::ShardedShape& mm_shape,
    const std::shared_ptr<MultiMeshStream>& stream,
    std::optional<StoreHandle> existing_store) const {
  if (has_store()) {
    return store();
  }

  if (VLOG_IS_ON(3)) {
    VLOG(3) << "converting PjRtBuffer to store: " << global_shape_
            << ", sharding=" << mm_shape;
  }

  VLOG(3) << "MultiMeshPjRtBuffer::ToStore: assembling from native "
             "buffer";

  TF_ASSIGN_OR_RETURN(auto* se_stream,
                      GetCachedStream(stream->mutable_backend(),
                                      device()->local_device_id().value()));

  TF_ASSIGN_OR_RETURN(auto buf_and_size,
                      GetPjRtBufferData(native_buffer(), se_stream));
  auto [buf, size] = std::move(buf_and_size);
  Shard mm_shard{.data = buf,
                 .local_device_id = device()->local_hardware_id().value(),
                 .size = size};

  return mm_client_->mutable_context()->AssembleShards(
      device_->local_device_id().value(), device_->global_device_id().value(),
      mm_shape, mm_shard, stream, std::move(existing_store));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> MultiMeshPjRtBuffer::SliceStore()
    const {
  void* sliced_buf = mm_client_->mutable_context()->SliceLocalShard(
      device_->local_device_id().value(), store());

  auto* se_client = dynamic_cast<PjRtStreamExecutorClient*>(base_client_);
  auto* cpu_client = dynamic_cast<TfrtCpuClient*>(base_client_);

  if (se_client == nullptr && cpu_client == nullptr) {
    return InternalStrCat(
        "client for MultiMeshPjRtBuffer is not a valid GPU or CPU client");
  }
  TF_ASSIGN_OR_RETURN(size_t size, this->GetOnDeviceSizeInBytes());
  std::shared_ptr<PjRtBuffer> native_buf;
  if (se_client) {
    se::DeviceMemoryBase mem(sliced_buf, size);
    auto base_event =
        std::make_shared<BufferSequencingEvent>(se_client->thread_pool());
    LocalDeviceState* device_state =
        &se_client->device_state(device_->local_hardware_id().value());
    se::Stream* stream = device_state->compute_stream();
    TF_ASSIGN_OR_RETURN(
        auto event,
        device_state->event_pool().ThenAllocateAndRecordEvent(stream));
    base_event->SetSequencingEvent(std::move(event), stream);

    std::unique_ptr<TrackedDeviceBuffer> buffer(new TrackedDeviceBuffer{
        device_,
        {RawSEDeviceMemory::Create(mem, device_->local_device_id(), nullptr)},
        {std::move(base_event)}});
    native_buf = std::make_unique<PjRtStreamExecutorBuffer>(
        on_device_shape(), std::move(buffer), se_client, device_,
        memory_space());
  } else {
    auto* cpu_device = dynamic_cast<TfrtCpuDevice*>(device_);
    if (cpu_device == nullptr) {
      return InternalStrCat(
          "device for MultiMeshPjRtBuffer is not a CPU device");
    }
    auto mem = CpuDeviceMemory::CreateForeignMemory(sliced_buf, size, [] {});
    auto tracked_buf =
        std::make_unique<TrackedCpuDeviceBuffer>(TrackedCpuDeviceBuffer(
            /*owns_buffers=*/true, std::move(mem),
            tsl::MakeAvailableAsyncValueRef<CpuEvent>()));
    native_buf = std::make_unique<TfrtCpuBuffer>(
        on_device_shape(), std::move(tracked_buf), cpu_client, cpu_device,
        memory_space());
  }

  auto shape = on_device_shape();
  return std::make_unique<MultiMeshPjRtBuffer>(
      std::move(native_buf), std::nullopt, shape, shape, mm_client_,
      base_client_, device_, memory_space(), name() + ".sliced");
}

std::pair<int64_t, int64_t> MultiMeshPjRtBuffer::ShardingDeviceRange() const {
  if (!sharding_.has_value() || sharding_->IsReplicated()) {
    return {0, base_client_->device_count()};
  }
  const auto& tile_assignment = sharding_->tile_assignment();
  size_t num_elements = tile_assignment.num_elements();
  return {tile_assignment.first(), tile_assignment.first() + num_elements};
}

std::pair<int64_t, int64_t> MultiMeshPjRtBuffer::SliceDeviceRange() const {
  // if replicated or if a scalar, this can be sliced across the entire machine
  if (!sharding_.has_value() || sharding_->IsReplicated() ||
      ShapeUtil::ElementsIn(global_shape_) == 1) {
    return {0, base_client_->device_count()};
  }
  return ShardingDeviceRange();
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
MultiMeshPjRtBuffer::SliceLocalShard() const {
  VLOG(3) << "MultiMeshPjRtBuffer::SliceLocalShards for sharding="
          << (sharding_.has_value() ? sharding_->ToString() : "replicated");

  if (has_native_buffer()) {
    VLOG(3) << "MultiMeshPjRtBuffer::SliceLocalShards: inputs are native PjRt "
               "buffers";
    return std::make_unique<MultiMeshPjRtBuffer>(
        shared_native_buffer(), std::nullopt, on_device_shape(),
        on_device_shape(), mm_client_, base_client_, device_, memory_space(),
        name() + ".sliced");
  }
  VLOG(3) << "MultiMeshPjRtBuffer::SliceLocalShards: input is a Zuku store";
  return SliceStore();
}

bool MultiMeshPjRtBuffer::HasValidShard() const {
  if (has_store()) {
    // this might be an empty dummy buffer for a sharded array
    return mm_client_->mutable_context()->HasLocalShard(store());
  }
  return true;
}

PjRtFuture<> MultiMeshPjRtBuffer::GetReadyFuture() {
  // As long as the data is not actively being retrieved we don't really care
  // whether the underlying buffer is ready -- once accessed/piped into
  // new task we will automatically synchronize it / wait for all active tasks
  return PjRtFuture<>(absl::OkStatus());
}

absl::StatusOr<size_t> MultiMeshPjRtBuffer::GetOnDeviceSizeInBytes() const {
  // just return approx for now
  return ShapeUtil::ByteSizeOfElements(local_shape_);
}

}  // namespace xla
