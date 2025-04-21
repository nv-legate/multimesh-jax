/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_MPMD_LOOP_H_
#define XLA_PJRT_LEGATE_MPMD_LOOP_H_

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "src/zuku/mesh.h"

namespace xla {

using CustomSchedule = std::vector<std::vector<std::pair<int64_t, std::string>>>;

// Encapsulates a Legate-managed loop that slices an input
// into microbatches and aggregates the results over the microbatch dimension.
struct LoopConfig {
  enum class Schedule {
    kFillDrain,
    k1F1B,
    kWavefront,
    kPrefetchWavefront,
    kCustom
  };

  // The number of iterations
  int64_t num_iterations;
  // A unique ID for the loop config
  int64_t unique_id;
  // The size of eachh microbatch slice
  int32_t microbatch_size;
  // The dimension of the sliced tensor being microbatched
  // This should correspond to a batch dimension
  // There are no safety checks to ensure this is a batch dimension
  int microbatch_dim;
  // The dimension of the dynamic slice that corresponds to the batch slice
  // This can be different from microbatch_dim if the slice is reshaped
  // to create optimally strided slices for data parallelism
  int slice_dim;
  // If a given task slices the microbatched input, the slice
  // offset will be given the HLO module as the given parameter number.
  std::optional<int64_t> slice_param_number{};
  // The type of pipeline schedule to follow
  Schedule schedule{Schedule::kFillDrain};
  // The amount of interleaving
  std::optional<int> interleave{};
  // The number of stages expected. Used in algorithms like 1F1B
  // to understand the structure of the loop
  std::optional<int> num_stages{};
  // The number of layers to unroll at a time. Only required for
  // the fill-drain (breadth-first) schedule
  std::optional<int> unrolling{};
  // The user specified custom pipeline schedule. Only required
  // for custom schedule option.
  std::optional<CustomSchedule> custom_schedule{};
};

// Encapsulates a device mesh that changes on each iteration.
// This can be used to e.g. load balance irregular layers
// by assigning that layer to different meshes for each microbatch.
class LoopDependentSubmesh {
 public:
  struct Config {
    int64_t task_mesh_size{-1};
    int64_t global_mesh_start{-1};
    int64_t global_mesh_stop{-1};
    bool reverse{false};
  };

  explicit LoopDependentSubmesh(Config cfg) : cfg_(cfg) {}

  zuku::DeviceList SubmeshForIteration(int64_t iteration) const;

  int64_t TaskMeshSize() const { return cfg_.task_mesh_size; }

  bool operator==(const LoopDependentSubmesh& rhs) const {
    return cfg_.task_mesh_size == rhs.cfg_.task_mesh_size &&
           cfg_.reverse == rhs.cfg_.reverse &&
           cfg_.global_mesh_start == rhs.cfg_.global_mesh_start &&
           cfg_.global_mesh_stop == rhs.cfg_.global_mesh_stop;
  }

 private:
  Config cfg_;
};

std::ostream& operator<<(std::ostream& os, const LoopDependentSubmesh& submesh);
std::ostream& operator<<(std::ostream& os,
                         const std::optional<LoopDependentSubmesh>& submesh);

// Creates a microbatch config from the Json string `text`. For debugging,
// the `name` is used for error messages.
absl::StatusOr<LoopConfig> GetMicrobatchConfig(const std::string& name,
                                               const std::string& text);

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_LOOP_H_
