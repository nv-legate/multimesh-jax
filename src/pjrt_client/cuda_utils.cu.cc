#include "xla/pjrt/legate/cuda_utils.h"

#include <iostream>

#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"

namespace {

void CheckError(CUresult result, const char* message) {
  if (result != CUDA_SUCCESS) {
    const char* error_string;
    auto error = cuGetErrorString(result, &error_string);
    if (error != CUDA_SUCCESS) {
      std::cerr << "Unknown error in cuda_utils::" << message << std::endl;
    } else {
      std::cerr << "Error in cuda_utils::" << message << ": " << error_string
                << std::endl;
    }
    abort();
  }
}

}  // namespace

namespace cuda_utils {

void CopyHostToDevice(void* dst, const void* src, size_t size) {
  auto result = cuMemcpyHtoD((CUdeviceptr)dst, src, size);
  CheckError(result, "CopyHostToDevice");
}

void CopyDeviceToDevice(void* dst, const void* src, size_t size) {
  auto result = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, size);
  CheckError(result, "CopyDeviceToDevice");
}

void CopyDeviceToHost(void* dst, const void* src, size_t size) {
  auto result = cuMemcpyDtoH(dst, (CUdeviceptr)src, size);
  CheckError(result, "cuMemcpyDtoH");
}

void Block() {
  auto result = cuCtxSynchronize();
  CheckError(result, "Block");
}

void* AllocateDeviceMemory(size_t size) {
  void* ptr;
  auto status = cuMemAlloc((CUdeviceptr*)&ptr, size);
  CheckError(status, "AllocateDeviceMemory");
  return ptr;
}

void SetPrimaryContext(int device_id) {
  CUdevice device;
  auto status = cuDeviceGet(&device, device_id);
  CheckError(status, "SetPrimaryContext");

  CUcontext ctx;
  status = cuDevicePrimaryCtxRetain(&ctx, device);
  CheckError(status, "SetPrimaryContext");

  status = cuCtxSetCurrent(ctx);
  CheckError(status, "SetPrimaryContext");
}

void Init() {
  auto status = cuInit(0);
  CheckError(status, "Init");
}

absl::Status RecordCuEvent(zuku::Stream* zs, stream_executor::Stream* stream) {
  stream_executor::gpu::GpuStreamHandle handle =
      stream_executor::gpu::AsGpuStreamValue(stream);
  zs->Record(reinterpret_cast<zuku::EventStream*>(handle));
  return absl::OkStatus();
}

std::string GpuDeviceName() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  return prop.name;
}

}  // namespace cuda_utils