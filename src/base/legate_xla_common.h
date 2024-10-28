#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdint.h>
#include <utility>
#include <variant>
#include <vector>

#include "src/zuku/shape.h"
#include "src/zuku/stream.h"

namespace legate_xla {

struct StartupConfig {
  std::optional<int> cpus;
  std::optional<int> gpus;
  std::optional<int64_t> fbmem;
  std::optional<int64_t> zcmem;
  std::optional<int64_t> sysmem;
  std::optional<std::string> network;
  bool kthreads{false};
  bool profile{false};
  std::vector<std::string> argv;
};

class TaskMemoryAllocator {
public:
  virtual void *Allocate(size_t size) = 0;

  virtual void Free(void *buf, size_t size) = 0;
};

struct BufferAllocation {
  void *buffer;
  size_t size;
};

struct InputOutputAliasConfig {
  std::unordered_map<int64_t, int64_t> param_to_output;
  std::unordered_map<int64_t, int64_t> output_to_param;
};

struct CompileConfig {
  int replica_count = 1;
  int num_partitions = 1;
  bool run_hlo_passes = true;
  int stream_executor_index = 0;
  TaskMemoryAllocator *allocator = nullptr;
  bool print_stats = false;
  bool erase_sharding = false;
  std::optional<int64_t> device_mem;
};

struct DeviceConfig {
  int64_t local_device_id = 0;
  int64_t global_device_id = 0;
  int64_t replica_count = 1;
  int64_t num_partitions = 1;
};

class DeviceAssignment {
public:
  explicit DeviceAssignment(const DeviceConfig &config,
                            zuku::DeviceList devices)
      : local_device_id_(config.local_device_id),
        global_device_id_(config.global_device_id),
        replica_count_(config.replica_count),
        num_partitions_(config.num_partitions), devices_(std::move(devices)) {}

  int64_t operator()(int replica, int partition) {
    return devices_[partition * replica_count_ + replica];
  }

  int64_t GlobalDeviceId() const { return global_device_id_; }

  int64_t LocalDeviceId() const { return local_device_id_; }

  int64_t ReplicaCount() const { return replica_count_; }

  int64_t NumPartitions() const { return num_partitions_; }

  int64_t GlobalDeviceId(int replica, int partition) const {
    return devices_[partition * replica_count_ + replica];
  }

private:
  int64_t local_device_id_;
  int64_t global_device_id_;
  int64_t replica_count_;
  int64_t num_partitions_;
  zuku::DeviceList devices_;
};

class LegateExecutable {
public:
  enum Platform {
    GPU = 0,
    CPU = 1,
  };

  virtual ~LegateExecutable() = default;
  /**
  * @brief
  *
  * @param run_id A unique ID identifying the legate task launch. This must be
  the same on all tasks that are part of the same launch.
  * @param inputs A list of input buffers matching the order of the input
  program shape
  * @param outputs A list of output buffers matching the order of the output
  program shape
  * @param allocator A local memory allocator for the current legate task
  * @param device_assignment The local device and global device ID array
  * @return Whether the execution ran successfully on the device
  */
  virtual std::optional<std::string>
  Execute(zuku::Stream *stream, uint64_t run_id,
          const std::vector<BufferAllocation> &inputs,
          const std::vector<BufferAllocation> &outputs,
          TaskMemoryAllocator *allocator,
          const DeviceAssignment &device_assignment, Platform platform,
          bool blocking) const = 0;

  virtual size_t LaunchSize() const = 0;

  virtual int ReplicaCount() const = 0;

  virtual int NumPartitions() const = 0;

  virtual std::string Name() const = 0;

  virtual std::optional<std::string>
  MemcpyHtoDAsync(void *dst, const void *src, size_t size,
                  int64_t local_device_id) const = 0;

  virtual std::optional<std::string>
  MemcpyDtoDAsync(void *dst, const void *src, size_t size, bool cpu,
                  int64_t local_device_id) const = 0;
};

class LegateStream {
public:
  virtual ~LegateStream() = default;

  virtual std::optional<std::string>
  MemcpyHtoDAsync(void *dst, const void *src, size_t size,
                  int64_t local_device_id) const = 0;

  virtual std::optional<std::string>
  MemcpyDtoDAsync(void *dst, const void *src, size_t size, bool cpu,
                  int64_t local_device_id) const = 0;
};

class LegateCompiler {
public:
  virtual ~LegateCompiler() = default;

  virtual void Compile(uint64_t run_id, const CompileConfig &config) = 0;

  virtual std::unique_ptr<LegateExecutable> MakeExecutable() = 0;

  virtual zuku::DeviceList MachineSlice() const = 0;

  virtual size_t LaunchSize() const = 0;

  virtual std::string Name() const = 0;

  virtual uint64_t HloId() const = 0;
};

class BufferAction {
public:
  virtual void Act(void *dst, int local_device_id) = 0;

  virtual ~BufferAction() = default;
};

struct ScalarArgument {
  using ValueVariant =
      std::variant<float, double, int32_t, int64_t, uint32_t, uint64_t>;
  ValueVariant value;
  int64_t parameter_number;
};

struct StoreHandleImpl;

struct StoreHandle {
  std::shared_ptr<StoreHandleImpl> impl;
  int64_t unique_id{-1};
  ~StoreHandle();
};

const zuku::Sharding &GetSharding(const StoreHandle &handle);
StoreHandle View(const StoreHandle &handle);
StoreHandle Child(const StoreHandle &handle);

struct BufferHandleImpl;
struct BufferHandle {
  std::shared_ptr<BufferHandleImpl> impl;
  ~BufferHandle();
};

static constexpr int64_t kTempMinAlignment = 4096;

inline int64_t AlignTempSize(int64_t size) {
  return ((size + kTempMinAlignment - 1) / kTempMinAlignment) *
         kTempMinAlignment;
}

void SetStartupConfig(StartupConfig cfg);

} // namespace legate_xla
