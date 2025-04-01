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

#include "xla/pjrt/legate/legate_buffer_action.h"

#include <memory>
#include <optional>
#include <variant>

#include "tsl/platform/logging.h"
#include "xla/pjrt/legate/cuda_utils.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"

namespace xla {

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
    WaitForBufferDefinitionEventsOnStream(*hold.buffer(), stream);
  } else {
    auto new_stream = se_device->local_device_state()->BorrowStreamFromPool();
    WaitForBufferDefinitionEventsOnStream(*hold.buffer(), new_stream.get());
    // this will cuStreamSynchronize when the new_stream destructor is called
  }

  const void* ptr = se_client_buffer->AsShapedBuffer()->root_buffer().opaque();
  size_t size = se_client_buffer->AsShapedBuffer()->root_buffer().size();

  return std::make_pair(ptr, size);
}

void BufferFromHostBufferAction::Act(void* dst, int local_device_id) {
  auto result = MakeBufferFromHostBuffer(dst, local_device_id);
  if (!result.ok()) {
    LOG(ERROR) << result.message();
  }
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
BufferFromHostBufferAction::MakeNativeBuffer(int device_id) {
  auto* local_device = client_->addressable_devices()[device_id];

  const void* data = std::holds_alternative<const void*>(data_)
                         ? std::get<const void*>(data_)
                         : std::get<std::vector<char>>(data_).data();

  TF_ASSIGN_OR_RETURN(auto* memory_space, local_device->default_memory_space());
  return client_->BufferFromHostBuffer(data, type_, dims_, byte_strides_,
                                       host_buffer_semantics_, nullptr,
                                       memory_space, nullptr);
}

absl::Status BufferFromHostBufferAction::MakeBufferFromHostBuffer(
    void* dst, int device_id) {
  TF_ASSIGN_OR_RETURN(auto device_src_buffer, MakeNativeBuffer(device_id));

  if (device_src_buffer->IsOnCpu()) {
    TF_ASSIGN_OR_RETURN(auto external_ref,
                        device_src_buffer->AcquireExternalReference());
    void* src_buf = external_ref->OpaqueDeviceMemoryDataPointer();
    TF_ASSIGN_OR_RETURN(auto size, device_src_buffer->GetOnDeviceSizeInBytes());
    ::memcpy(dst, src_buf, size);
  } else {
    TF_ASSIGN_OR_RETURN(auto src, GetPjRtBufferData(device_src_buffer.get()));
    auto [src_device_ptr, src_device_size] = std::move(src);

    cuda_utils::CopyDeviceToDevice(dst, src_device_ptr, src_device_size);
  }

  if (on_done_with_host_buffer_) {
    (std::move(on_done_with_host_buffer_))();
  }

  return absl::OkStatus();
}

}  // namespace xla