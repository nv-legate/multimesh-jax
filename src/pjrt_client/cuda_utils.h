#ifndef XLA_PJRT_CUDA_UTILS_H_
#define XLA_PJRT_CUDA_UTILS_H_

#include <string>

#include "src/zuku/stream.h"
#include "xla/stream_executor/stream.h"

namespace cuda_utils {

void CopyHostToDevice(void* dst, const void* src, size_t size);

void CopyDeviceToDevice(void* dst, const void* src, size_t size);

void CopyDeviceToHost(void* dst, const void* src, size_t size);

void DevicePrintf(const void* buffer, size_t num_elems);

void Block();

void Init();

void SetPrimaryContext(int device_id);

void* AllocateDeviceMemory(size_t size);

absl::Status RecordCuEvent(zuku::Stream* zs, stream_executor::Stream* stream);

std::string GpuDeviceName();

}  // namespace cuda_utils

#endif
