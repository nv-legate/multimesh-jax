/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_MPMD_PARTITION_H_
#define XLA_PJRT_LEGATE_MPMD_PARTITION_H_

#include <optional>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/legate_computation.h"
#include "xla/pjrt/legate/mpmd_store.h"
#include "xla/pjrt/legate/scalar_argument.h"
#include "xla/pjrt/pjrt_executable.h"

namespace xla {

// A single task from an MPMD decomposition that corresponds
// to a single SPMD subcomputation on a submesh

struct SpmdModule {
  bool loop_increment{false};
  std::unique_ptr<HloModule> module;
  std::optional<int64_t> slice_param_number;
  absl::flat_hash_set<zuku::DeviceList> device_meshes;
};

struct SpmdHloModuleTask {
  std::shared_ptr<SpmdModule> module;
  std::shared_ptr<LegateCompiler> compiler;
  zuku::DeviceList device_assignment;
  std::vector<Store> inputs;
  std::vector<Store> outputs;
  bool loop;
  std::vector<ScalarArgument> scalars;
};

struct Reshard {
  Store input;
  Store output;
};

struct FreeStore {
  Store store;
};

struct CreateStore {
  Store store;
};

struct MpmdOperation {
  std::variant<SpmdHloModuleTask, Reshard, FreeStore, CreateStore> op;
};

struct MpmdPartitionConfig {
  bool use_task_fusion{true};
  std::optional<int64_t> replicated_parameter_num_elements_cutoff{std::nullopt};
  bool run_simplification_passes{true};
  bool only_fuse_loop_tasks{false};
  bool hoist_loop_convert{false};
  bool remove_hoisted_reduces{false};
  std::optional<int64_t> recompute_from_arguments_if_cost_less_than{
      std::nullopt};
};

// Partitions a fully partitioned HLO module into tasks with task HLO modules.
// The `module` should be in "grouped" form with all tasks as calls
// in the entry computation. Returns a vector of tasks and the number
// of unique temporaries needed between tasks.
absl::StatusOr<std::tuple<std::vector<MpmdOperation>,
                          std::vector<std::shared_ptr<SpmdModule>>,
                          std::vector<Store>, std::vector<Store>>>
MpmdPartitionIntoTasks(HloModule* module, const HloPartition& partition,
                       const ExecutableBuildOptions& options);

// Partitions a raw input HLO module into multiple tasks based on task
// definitions in the module proto.
//
// `options`, `argument_layout_pointers`, `executable_layout` are the standard
// arguments passed to PjRtClient::Compile. The `config` has multiple
// options for tuning how the module is partitioned and optimized.
// Returns a scheduled-order vector operations, the vector of unique
// SPMD modules, and the vector of unique temporaries.
absl::StatusOr<std::tuple<std::vector<MpmdOperation>,
                          std::vector<std::shared_ptr<SpmdModule>>,
                          std::vector<Store>, std::vector<Store>>>
MpmdPartition(const HloModuleProto& proto, const CompileOptions& options,
              const std::vector<const Shape*>& argument_layout_pointers,
              const Shape& executable_layout,
              const MpmdPartitionConfig& config = {});

}  // namespace xla

extern "C" void SetHostOffloadMinSize(int64_t size);

#endif  // XLA_PJRT_LEGATE_MPMD_PARTITION_H_
