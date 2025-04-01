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

#ifndef XLA_PJRT_LEGATE_BUFFER_ACTION_H_
#define XLA_PJRT_LEGATE_BUFFER_ACTION_H_

#include "xla/pjrt/pjrt_client.h"

namespace xla {

class BufferAction {
 public:
  virtual void Act(void* dst, int local_device_id) = 0;

  virtual ~BufferAction() = default;
};

/**
 @brief Encapsulates an action callback to a Legate task that creates a stream
 executor device buffer from a host buffer
 */
class BufferFromHostBufferAction : public BufferAction {
 public:
  BufferFromHostBufferAction(
      std::variant<const void*, std::vector<char>> data, PrimitiveType type,
      absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      PjRtClient::HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtClient* client)
      : data_(data),
        type_(type),
        dims_{dims.begin(), dims.end()},
        byte_strides_(std::nullopt),
        host_buffer_semantics_(host_buffer_semantics),
        on_done_with_host_buffer_(std::move(on_done_with_host_buffer)),
        client_(client) {
    if (byte_strides.has_value()) {
      byte_strides_storage_ = {byte_strides->begin(), byte_strides->end()};
      byte_strides_ = absl::MakeSpan(byte_strides_storage_);
    }
  }

  ~BufferFromHostBufferAction() override = default;

  void Act(void* dst, int local_device_id) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> MakeNativeBuffer(int device_id);

 private:
  absl::Status MakeBufferFromHostBuffer(void* dst, int device_id);

  std::variant<const void*, std::vector<char>> data_;
  PrimitiveType type_;
  std::vector<int64_t> dims_;
  std::vector<int64_t> byte_strides_storage_;
  std::optional<absl::Span<int64_t const>> byte_strides_;
  std::unique_ptr<PjRtBuffer> device_src_buffer_;
  PjRtClient::HostBufferSemantics host_buffer_semantics_;
  absl::AnyInvocable<void() &&> on_done_with_host_buffer_;
  PjRtClient* client_;
};

absl::StatusOr<std::pair<const void*, size_t>> GetPjRtBufferData(
    PjRtBuffer* buffer, se::Stream* stream = nullptr);

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_BUFFER_ACTION_H_
