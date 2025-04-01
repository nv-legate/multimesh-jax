#ifndef XLA_PJRT_LEGATE_UTILS_H_
#define XLA_PJRT_LEGATE_UTILS_H_

#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/legate/allocator.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/backend.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

absl::StatusOr<se::Stream*> GetCachedStream(Backend* backend,
                                            int32_t device_id);

absl::StatusOr<se::Stream*> GetCachedStream(se::StreamExecutor* se,
                                            int32_t device_id);

void ClearCachedStreams();

std::shared_ptr<DistributedRuntimeClient> LegateRuntimeClient();
class TaskDeviceMemoryAllocator : public se::DeviceMemoryAllocator {
 public:
  // Parameter platform indicates which platform the allocator allocates memory
  // on. Must be non-null.
  explicit TaskDeviceMemoryAllocator(int device_ordinal,
                                     TaskMemoryAllocator* allocator,
                                     se::Stream* stream, Backend* backend);

  absl::StatusOr<se::OwningDeviceMemory> Allocate(
      int device_ordinal, uint64_t size, bool retry_on_failure,
      int64_t memory_space) override;

  absl::Status Deallocate(int device_ordinal,
                          se::DeviceMemoryBase mem) override;

  // Returns a stream pointer on which it is always safe to access memory
  // allocated by this allocator. It is not necessary to use the returned stream
  // though, as clients may have additional information letting them safely use
  // a different stream.
  absl::StatusOr<se::Stream*> GetStream(int device_ordinal) override;

 private:
  TaskMemoryAllocator* allocator_;
  int device_ordinal_;
  se::Stream* stream_;
};

class StreamWrapper {
 public:
  StreamWrapper(uint64_t run_id, int device_ordinal, se::Stream* stream,
                DeviceAssignment device_assignment, xla::Backend* backend,
                TaskMemoryAllocator* allocator = nullptr);

  int64_t ComputeTimeNs() const { return execution_profile_.compute_time_ns(); }

  xla::Backend* Backend() const { return backend_; }

  se::DeviceMemoryAllocator* MemoryAllocator();

  ServiceExecutableRunOptions* RunOptions() { return &service_run_options_; }

  se::Stream* XlaStream() { return stream_; }

  const xla::DeviceAssignment* DeviceAssignment() {
    return &device_assignment_;
  }

 private:
  se::Stream* stream_;
  xla::DeviceAssignment device_assignment_;
  xla::Backend* backend_;
  std::optional<TaskDeviceMemoryAllocator> allocator_;
  ServiceExecutableRunOptions service_run_options_;
  ExecutionProfile execution_profile_;
};

absl::Status InitDistributedRuntimeParams(
    int num_procs, int node_id, int gpus_per_node,
    std::shared_ptr<KeyValueStoreInterface> kv_store);

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_UTILS_H_
