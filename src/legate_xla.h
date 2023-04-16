#pragma once

#include <memory>
#include <stdint.h>
#include <utility>
#include <vector>
#include <functional>

namespace legate_xla {

class TaskMemoryAllocator {
 public:
  virtual void* Allocate(size_t size) = 0;

  virtual void Free(void* buf, size_t size) = 0;
};

struct BufferAllocation {
  void* buffer;
  size_t size;
};

struct CompileConfig {
  int replica_count              = 1;
  int num_partitions             = 1;
  bool run_hlo_passes            = true;
  int stream_executor_index      = 0;
  TaskMemoryAllocator* allocator = nullptr;
};

struct DeviceConfig {
  int local_device_id = 0;
  int replica_count   = 1;
  int num_partitions  = 1;
};

class DeviceAssignment {
 public:
  DeviceAssignment(const DeviceConfig& config)
    : local_device_id_(config.local_device_id),
      replica_count_(config.replica_count),
      num_partitions_(config.num_partitions),
      global_device_ids_(config.replica_count * config.num_partitions, /*fill_value=*/-1)
  {
  }

  int& operator()(int replica, int partition)
  {
    return global_device_ids_[partition * replica_count_ + replica];
  }

  int LocalDeviceId() const { return local_device_id_; }

  int ReplicaCount() const { return replica_count_; }

  int NumPartitions() const { return num_partitions_; }

  int GlobalDeviceId(int replica, int partition) const
  {
    return global_device_ids_[partition * replica_count_ + replica];
  }

 private:
  int local_device_id_;
  int replica_count_;
  int num_partitions_;
  std::vector<int> global_device_ids_;
};


class LegateExecutable {
 public:
  /**
  * @brief
  *
  * @param run_id A unique ID identifying the legate task launch. This must be the same
                  on all tasks that are part of the same launch.
  * @param inputs A list of input buffers matching the order of the input program shape
  * @param outputs A list of output buffers matching the order of the output program shape
  * @param allocator A local memory allocator for the current legate task
  * @param device_assignment The local device and global device ID array
  * @return Whether the execution ran successfully on the device
  */
  virtual bool Execute(uint64_t run_id,
                      const std::vector<BufferAllocation>& inputs,
                      const std::vector<BufferAllocation>& outputs,
                      TaskMemoryAllocator* allocator,
                      const DeviceAssignment& device_assignment) const = 0;

};

class LegateCompiler {
 public:
  virtual void Compile(uint64_t run_id, const CompileConfig& config) = 0;

  virtual std::unique_ptr<LegateExecutable> MakeExecutable() = 0;
};


class StoreHandle {};

enum class SupportedType {
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

struct Shape {
  SupportedType type;
  std::vector<int> dims;
};

void CreateCompileTask(LegateCompiler* compiler);

void CreateExecuteTask(LegateExecutable* executable,
  const std::vector<StoreHandle*>& inputs,
  const std::vector<StoreHandle*>& outputs);

StoreHandle* CreateStore(const legate_xla::Shape& shape);

StoreHandle* CreateStoreFromHostBuffer(
  const legate_xla::Shape& shape, const void* data, std::function<void()> on_done);

void InitLegate();

}

extern "C" std::unique_ptr<legate_xla::LegateCompiler> GetLegateCompilerFromHloProtoText(
  const std::string& hlo_string,
  const std::string& platform_name,
  int replica_count, int num_partitions);
