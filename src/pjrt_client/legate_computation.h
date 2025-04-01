#ifndef XLA_PJRT_LEGATE_LEGATE_COMPILATION_H_
#define XLA_PJRT_LEGATE_LEGATE_COMPILATION_H_

#include <memory>

#include "allocator.h"
#include "src/zuku/stream.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/legate/legate_device_assignment.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/backend.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {

struct LegateCompileConfig {
  int replica_count = 1;
  int num_partitions = 1;
  bool run_hlo_passes = true;
  bool run_backend = true;
  int stream_executor_index = 0;
  TaskMemoryAllocator* allocator = nullptr;
  bool print_stats = false;
  bool erase_sharding = false;
  std::optional<int64_t> device_mem;
};

class LegateXlaComputation {
 public:
  static absl::StatusOr<std::unique_ptr<LegateXlaComputation>> Create(
      std::unique_ptr<HloModule> module, Shape root_shape, Backend* backend,
      PjRtClient* client);

  ~LegateXlaComputation() = default;

  absl::Status Compile(uint64_t run_id, const LegateCompileConfig& config);

  const Backend* backend() const { return backend_; }

  PjRtClient* client() { return client_; }

  Backend* mutable_backend() { return backend_; }

  const HloModule& module() const { return *input_module_; }

  const HloModule& optimized_module() const;

  const Executable& executable() const { return *executable_; }

  void SetExecutable(std::unique_ptr<Executable> executable) {
    executable_ = std::move(executable);
  }

  Executable* mutable_executable() { return executable_.get(); }

  absl::StatusOr<std::shared_ptr<HloModule>> GetHloModule() const;

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules() const;

  void MemorySummary(int vlog);

  void CostAnalysis(int vlog);

  uint64_t HloId() const { return hlo_id_; }

  const Shape& RootShape() const { return root_shape_; }

  int ReplicaCount() const {
    return executable_->module_config().replica_count();
  }

  int NumPartitions() const {
    return executable_->module_config().num_partitions();
  }

  bool Concurrent() const { return concurrent_; }

  bool CompiledLocally() const { return executable_ != nullptr; }

  int64_t TempRequired() const;

  std::string Name() const { return input_module_->name(); }

  size_t LaunchSize() const { return launch_size_; }

  std::optional<int64_t> OutputAlias(int64_t root_index) const;

 private:
  absl::Status SetupOutputInputAlias();

  LegateXlaComputation(std::unique_ptr<HloModule> module, Backend* backend,
                       PjRtClient* client, size_t launch_size, Shape root_shape)
      : hlo_id_(module->unique_id()),
        input_module_(std::move(module)),
        opt_module_(input_module_->Clone("")),
        backend_(backend),
        client_(client),
        launch_size_(launch_size),
        concurrent_(
            true),  // task starts as concurrent until we can prove otherwise
        root_shape_(std::move(root_shape)) {}

  uint64_t hlo_id_;

  std::unique_ptr<Executable> executable_;

  std::unique_ptr<HloModule> input_module_;

  std::unique_ptr<HloModule> opt_module_;

  PjRtClient* client_;

  absl::flat_hash_map<int64_t, int64_t> output_input_alias_;

  size_t launch_size_;

  std::optional<int64_t> temp_required_;

  Backend* backend_;

  Shape root_shape_;

  bool concurrent_;
};

class LegateStream {
 public:
  explicit LegateStream(Backend* backend) : backend_(backend) {}

  ~LegateStream() = default;

  std::optional<std::string> MemcpyHtoDAsync(void* dst, const void* src,
                                             size_t size,
                                             int64_t local_device_id) const;

  std::optional<std::string> MemcpyDtoDAsync(void* dst, const void* src,
                                             size_t size, bool cpu,
                                             int64_t local_device_id) const;

  Backend* mutable_backend() { return backend_; }

 private:
  Backend* backend_;
};

class LegateExecutable {
 public:
  enum Platform {
    GPU = 0,
    CPU = 1,
  };

  explicit LegateExecutable(
      const std::shared_ptr<LegateXlaComputation>& computation);

  std::optional<std::string> Execute(
      zuku::Stream* zs, uint64_t run_id,
      const std::vector<se::DeviceMemoryBase>& inputs,
      const std::vector<se::DeviceMemoryBase>& outputs,
      TaskMemoryAllocator* allocator,
      const LegateDeviceAssignment& device_assignment, int64_t num_local,
      Platform platform, bool blocking) const;

  size_t LaunchSize() const { return computation_->LaunchSize(); }

  int ReplicaCount() const { return computation_->ReplicaCount(); }

  int NumPartitions() const { return computation_->NumPartitions(); }

  std::string Name() const { return computation_->Name(); }

  std::optional<std::string> MemcpyHtoDAsync(void* dst, const void* src,
                                             size_t size,
                                             int64_t local_device_id) const;

  std::optional<std::string> MemcpyDtoDAsync(void* dst, const void* src,
                                             size_t size, bool cpu,
                                             int64_t local_device_id) const;

 private:
  std::shared_ptr<LegateXlaComputation> computation_;
};

class LegateCompiler : public LegateStream {
 public:
  static absl::StatusOr<std::unique_ptr<LegateCompiler>> Create(
      std::unique_ptr<HloModule> module, Shape root_shape, Backend* backend,
      PjRtClient* client);

  ~LegateCompiler() = default;

  const HloModule& optimized_module() const {
    return computation_->optimized_module();
  }

  const Shape& RootShape() const { return computation_->RootShape(); }

  void Compile(uint64_t run_id, const LegateCompileConfig& config);

  size_t LaunchSize() const { return computation_->LaunchSize(); }

  const HloModule& module() const { return computation_->module(); }

  se::Stream* GetStream(int64_t local_device_id);

  void SetExecutable(std::unique_ptr<Executable> executable) {
    compilation_finished_ = true;
    computation_->SetExecutable(std::move(executable));
  }

  bool CompiledLocally() const { return computation_->CompiledLocally(); }

  bool Concurrent() const { return computation_->Concurrent(); }

  int64_t TempRequired() const { return computation_->TempRequired(); }

  Backend* mutable_backend() { return computation_->mutable_backend(); }

  std::optional<int64_t> OutputAlias(int64_t root_index) const {
    return computation_->OutputAlias(root_index);
  }

  std::unique_ptr<LegateExecutable> MakeExecutable();

  uint64_t HloId() const { return computation_->HloId(); }

  std::string Name() const { return computation_->Name(); }

  std::map<int, int>& OutputInputAlias();

  void Block();

 private:
  explicit LegateCompiler(std::unique_ptr<LegateXlaComputation> comp);

  std::shared_ptr<LegateXlaComputation> computation_;

  std::condition_variable condition_;

  std::optional<int64_t> temp_required_;

  std::mutex mutex_;
  bool compilation_finished_ = false;

  Shape root_shape_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_LEGATE_COMPILATION_H_