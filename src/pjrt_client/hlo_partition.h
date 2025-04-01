#ifndef XLA_PJRT_LEGATE_HLO_PARTITION_H_
#define XLA_PJRT_LEGATE_HLO_PARTITION_H_

#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/legate/logical_sharding_context.h"

namespace xla {

// Returns whether a given HLO module contains any Legate custom calls.
bool ContainsLegateCustomCall(const HloModuleProto& proto);

class HloPartition {
 public:
  struct ColorConfig {
    std::string name;
    zuku::DeviceList devices;
    std::shared_ptr<LogicalShardingContext> logical_sharding_context;
  };

  // Public factory function for creating a partition of an HLO module.
  // This function should return a fully partitioned HLO module with all tasks
  // created.
  static absl::StatusOr<HloPartition> Create(HloModule* module,
                                             zuku::DeviceList devices);

  // Loops through all registered ImplicitTaskMatcher objects and
  // finds one matching metadata of the instruction. Returns std::nullopt
  // if the instruction has not metadata.op_name or there is not matching
  // implicit task register.
  absl::StatusOr<std::optional<std::string>> ComputeMetadataNameColor(
      HloInstruction* instruction);

  std::optional<std::string> FindColor(const zuku::DeviceList& devices) const;

  bool HasColor(const std::string& color) const;

  absl::StatusOr<std::string> FindOrAllocateColor(
      std::string name, zuku::DeviceList devices,
      std::shared_ptr<LogicalShardingContext> context);

  absl::StatusOr<std::string> FindOrAllocateDefaultColor(
      std::optional<std::string> name = std::nullopt);

  absl::StatusOr<std::string> AllocateLoopIncrementColor();

  bool IsLoopIncrementColor(const std::string& color) const;

  bool SameMesh(const HloInstruction* lhs, const HloInstruction* rhs) const;

  bool EquivalentMesh(const std::string& lhs_color,
                      const std::string& rhs_color) const;

  int64_t NumDevicesForInstruction(const HloInstruction* instruction) const;

  absl::StatusOr<std::string> FindOrAllocateGlobalColor(std::string name);

  // Allocates a new color for the unique `name` corresponding to the given
  // `devices` submesh. Multiple colors with different names can be allocated
  // for a given device mesh. An optional autosharding context can be assigned
  // to the color fo translating logical axis names to physical shardings.
  absl::StatusOr<std::string> AllocateColor(
      std::string name, zuku::DeviceList devices,
      std::shared_ptr<LogicalShardingContext> context = nullptr);

  absl::Status AllocateColor(
      std::string name,
      std::shared_ptr<LogicalShardingContext> context = nullptr);

  const zuku::DeviceList& DefaultDevices() const {
    return default_config_.devices;
  }

  const ColorConfig& DefaultConfig() const { return default_config_; }

  zuku::DeviceList DefaultPerTaskMesh(const std::string& color) const;

  zuku::DeviceList DefaultPerTaskMesh(HloInstruction* instruction) const;

  const zuku::DeviceList& DevicesForColor(const std::string& color) const;

  const zuku::DeviceList& DevicesForInstruction(
      HloInstruction* instruction) const;

  const ColorConfig& ConfigForColor(const std::string& color) const;

  const zuku::DeviceList& Devices() const { return devices_; }

  std::shared_ptr<LogicalShardingContext> GetLogicalShardingContext(
      const std::string& color) const;

  absl::Status Recolor(
      const absl::flat_hash_map</*new=*/std::string, /*old*/ std::string>&
          color_map);

  int64_t TotalDevices() const { return devices_.size(); }

 private:
  HloPartition(
      HloModule* module, zuku::DeviceList global_devices,
      zuku::DeviceList default_per_color_devices,
      std::shared_ptr<LogicalShardingContext> default_logical_sharding_context)
      : devices_(global_devices),
        default_config_({.name = "default",
                         .devices = default_per_color_devices,
                         .logical_sharding_context =
                             std::move(default_logical_sharding_context)}) {}

  zuku::DeviceList devices_;

  ColorConfig default_config_;

  int64_t max_color_{0};

  // needs to be ordered by color
  absl::flat_hash_map<std::string, ColorConfig> colors_;

  absl::flat_hash_map<zuku::DeviceList, absl::InlinedVector<std::string, 2>>
      device_list_to_colors_;
};

void ValidateModuleMetadata(HloModule* module);

}  // namespace xla

extern "C" void RegisterMetadataNameMatcher(std::string matcher,
                                            std::optional<std::string> name);

// C/Python binding for registering tasks based on metadata names.
// `matcher` is the regular expression used to match a task.
// The regex group for `matcher` should produce a unique task name.
// `devices` is the device submesh associated with the task.
// `dims` is the mesh shape (x,y,z) matching the device list.
// `axes` are string names associated with each physical dimension in `dims`
// `logical_axes` are (logical, device) pairs as used in pjit/T5x
// e.g. ('batch', 'x') says that the logical batch dimension should be
// matched to the 'x' physical dimension
extern "C" void RegisterMetadataNameTask(
    std::string matcher, std::vector<int64_t> devices,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes,
    int64_t fusion_color = 0,
    std::optional<int64_t> loop_submesh_size = std::nullopt,
    bool loop_submesh_reverse = false);

// C/Python binding for registering tasks based on metadata names.
// Rather than use a fixed device argument,
// this dynamically generates a device list based on `matcher`.
// The regex group from `matcher` is passed to `device_factory` to produce
// a device list and then forwarded to `RegisterMetadataNameTask`.  For example,
// layer_0 could be matched and mapped to Devices 0-3, while layer_1
// could be matched and mapped to Devices 4-7.
extern "C" void RegisterMetadataNameTaskWithFactory(
    std::string matcher,
    std::function<std::vector<int64_t>(const std::string& task)> device_factory,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes,
    int64_t fusion_color = 0);

// Turn on/off whether implicit tasks should be matched and created
extern "C" void SetEnableMetadataNameTasks(bool flag);

// Clear all previously registerd implicit tasks
extern "C" void ClearMetadataNameTasks();

extern "C" void EnableLegateRecomputation(bool enable);

extern "C" void SetHostOffloadMinReuseDistance(int64_t reuse_distance);

#endif  // XLA_PJRT_LEGATE_HLO_PARTITION_H_