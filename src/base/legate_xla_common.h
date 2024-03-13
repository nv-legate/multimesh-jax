#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <optional>
#include <stdint.h>
#include <utility>
#include <variant>
#include <vector>

namespace legate_xla {

class TaskMemoryAllocator {
public:
  virtual void *Allocate(size_t size) = 0;

  virtual void Free(void *buf, size_t size) = 0;
};

struct BufferAllocation {
  void *buffer;
  size_t size;
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
  int local_device_id = 0;
  int global_device_id = 0;
  int replica_count = 1;
  int num_partitions = 1;
};

class DeviceAssignment {
public:
  explicit DeviceAssignment(const DeviceConfig &config)
      : local_device_id_(config.local_device_id),
        global_device_id_(config.global_device_id),
        replica_count_(config.replica_count),
        num_partitions_(config.num_partitions),
        global_device_ids_(config.replica_count * config.num_partitions,
                           /*fill_value=*/-1) {}

  int &operator()(int replica, int partition) {
    return global_device_ids_[partition * replica_count_ + replica];
  }

  int GlobalDeviceId() const { return global_device_id_; }

  int LocalDeviceId() const { return local_device_id_; }

  int ReplicaCount() const { return replica_count_; }

  int NumPartitions() const { return num_partitions_; }

  int GlobalDeviceId(int replica, int partition) const {
    return global_device_ids_[partition * replica_count_ + replica];
  }

private:
  int local_device_id_;
  int global_device_id_;
  int replica_count_;
  int num_partitions_;
  std::vector<int> global_device_ids_;
};

class LegateExecutable {
public:
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
  Execute(uint64_t run_id, const std::vector<BufferAllocation> &inputs,
          const std::vector<BufferAllocation> &outputs,
          TaskMemoryAllocator *allocator,
          const DeviceAssignment &device_assignment, bool cpu) const = 0;

  virtual std::pair<int, int> MachineSlice() const = 0;

  virtual size_t LaunchSize() const = 0;

  virtual int ReplicaCount() const = 0;

  virtual int NumPartitions() const = 0;

  virtual std::string Name() const = 0;
};

class LegateCompiler {
public:
  virtual ~LegateCompiler() = default;

  virtual void Compile(uint64_t run_id, const CompileConfig &config) = 0;

  virtual std::unique_ptr<LegateExecutable> MakeExecutable() = 0;

  virtual std::pair<int, int> MachineSlice() const = 0;

  virtual size_t LaunchSize() const = 0;

  virtual std::string Name() const = 0;

  virtual uint64_t HloId() const = 0;
};

template <class T> class TaskArgHold {
public:
  explicit TaskArgHold(const std::shared_ptr<T> &arg) : arg_(arg) {}

  T *operator->() const { return arg_.get(); }

  T *get() const { return arg_.get(); }

  bool release(int num_total_holds) {
    int num_done = num_finished_.fetch_add(1);
    return num_done == (num_total_holds - 1);
  }

private:
  std::shared_ptr<T> arg_;
  std::atomic<int> num_finished_{0};
};

template <class T> TaskArgHold<T> *Hold(const std::shared_ptr<T> &arg) {
  return new TaskArgHold{arg};
}

template <class T> void Release(TaskArgHold<T> *hold, int num_total_holds) {
  if (hold->release(num_total_holds)) {
    delete hold;
  }
}

class BufferAction {
public:
  virtual void Act(void *dst, int local_device_id) = 0;

  virtual ~BufferAction() = default;
};

enum class SupportedType {
  PRED,
  F16,
  BF16,
  F32,
  F64,
  S8,
  S16,
  S32,
  S64,
  U8,
  U16,
  U32,
  U64,
  C64,
  C128,
};

inline constexpr int64_t kMaxScalarArguments = 64;

struct ScalarArgument {
  using ValueVariant =
      std::variant<float, double, int32_t, int64_t, uint32_t, uint64_t>;
  ValueVariant value;
  int64_t parameter_number;
};

struct Shape {
  SupportedType type;
  std::vector<int64_t> dims;
  std::optional<std::vector<int64_t>> tile_shape;
  int64_t replicated{1};
  size_t num_tiles{1};
};

bool operator==(const Shape &lhs, const Shape &rhs);

struct Tile {
  std::vector<int64_t> origin;
  std::vector<int64_t> dims;
};

struct StoreHandleImpl;

struct StoreHandle {
  std::shared_ptr<StoreHandleImpl> impl;
  int64_t unique_id{-1};
  ~StoreHandle();
};

class TaskFuture {
public:
  explicit TaskFuture(int64_t num_tasks)
      : ready_(false), num_pending_{num_tasks} {}

  void Wait();

  int64_t Signal();

private:
  bool ready_;
  std::atomic<int64_t> num_pending_;
  std::condition_variable cv_;
  std::mutex m_;
};

struct StoreFuture {
  std::unique_ptr<TaskFuture> future{nullptr};
  legate_xla::StoreHandle store;
};

std::ostream &operator<<(std::ostream &os, const Shape &shape);

} // namespace legate_xla
