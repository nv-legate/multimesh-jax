#include "xla/pjrt/legate/mpmd_loop.h"

#include <ostream>

#include "absl/container/flat_hash_map.h"
#include "src/zuku/mesh.h"
#include "xla/pjrt/legate/json_utils.h"

namespace xla {

zuku::DeviceList LoopDependentSubmesh::SubmeshForIteration(
    int64_t iteration) const {
  const int64_t machine_size = cfg_.global_mesh_stop - cfg_.global_mesh_start;
  const int64_t num_slices = machine_size / cfg_.task_mesh_size;
  const int64_t stage = iteration % num_slices;
  const int64_t offset = stage * cfg_.task_mesh_size;
  if (cfg_.reverse) {
    const int64_t start = cfg_.global_mesh_stop - offset - cfg_.task_mesh_size;
    return zuku::DeviceList::Create(start, cfg_.task_mesh_size);
  }
  return zuku::DeviceList::Create(offset, cfg_.task_mesh_size);
}

absl::StatusOr<LoopConfig> GetMicrobatchConfig(const std::string& name,
                                               const std::string& text) {
  TF_ASSIGN_OR_RETURN(Json::Value json, GetJsonValue(text.data(), text.size()));

  LoopConfig defaults;

  TF_ASSIGN_OR_RETURN(const int num_microbatches,
                      GetTaskValue<int>(json, name, "num_microbatches"));
  TF_ASSIGN_OR_RETURN(const int size, GetTaskValue<int>(json, name, "size"));
  TF_ASSIGN_OR_RETURN(const int slice_dim,
                      GetTaskValue<int>(json, name, "slice_dim"));
  TF_ASSIGN_OR_RETURN(
      const int batch_dim,
      GetOptionalTaskValue<int>(json, name, "batch_dim", slice_dim));
  TF_ASSIGN_OR_RETURN(std::optional<int> unrolling,
                      GetOptionalTaskValue<int>(json, name, "unrolling"));
  TF_ASSIGN_OR_RETURN(
      std::string schedule_string,
      GetOptionalTaskValue<std::string>(json, name, "schedule", ""));
  LoopConfig::Schedule schedule = defaults.schedule;

  static const absl::flat_hash_map<std::string, LoopConfig::Schedule>
      kScheduleTypes = {
          {"1f1b", LoopConfig::Schedule::k1F1B},
          {"fill-drain", LoopConfig::Schedule::kFillDrain},
          {"breadth-first", LoopConfig::Schedule::kFillDrain},
          {"gpipe", LoopConfig::Schedule::kFillDrain},
          {"wavefront", LoopConfig::Schedule::kWavefront},
          {"prefetch-wavefront", LoopConfig::Schedule::kPrefetchWavefront}};

  if (!schedule_string.empty()) {
    auto iter = kScheduleTypes.find(schedule_string);
    if (iter == kScheduleTypes.end()) {
      std::string error_message =
          "Bad schedule type given to microbatch. Allow schedules are: ";
      for (const auto& [name, value] : kScheduleTypes) {
        absl::StrAppend(&error_message, ", ", name);
      }
      return InvalidArgumentStrCat(error_message);
    }
    schedule = iter->second;
  }

  TF_ASSIGN_OR_RETURN(std::optional<int> interleave,
                      GetOptionalTaskValue<int>(json, name, "interleave"));

  TF_ASSIGN_OR_RETURN(std::optional<int> num_stages,
                      GetOptionalTaskValue<int>(json, name, "num_stages"));

  return LoopConfig{
      .num_iterations = num_microbatches,
      .microbatch_size = size,
      .microbatch_dim = batch_dim,
      .slice_dim = slice_dim,
      .schedule = std::move(schedule),
      .interleave = interleave,
      .num_stages = num_stages,
      .unrolling = unrolling,
  };
}

std::ostream& operator<<(std::ostream& os,
                         const LoopDependentSubmesh& submesh) {
  os << "submesh[size=" << submesh.TaskMeshSize()
     << ",first=" << submesh.SubmeshForIteration(0) << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const std::optional<LoopDependentSubmesh>& submesh) {
  if (submesh.has_value()) {
    os << *submesh;
  } else {
    os << "no-submesh";
  }
  return os;
}

}  // namespace xla