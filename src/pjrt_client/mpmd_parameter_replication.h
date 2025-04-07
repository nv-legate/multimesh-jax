
/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_MPMD_PARAMETER_REPLICATION_H_
#define XLA_PJRT_LEGATE_MPMD_PARAMETER_REPLICATION_H_

#include <cstdint>

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"

namespace xla {

// Loops through and finds small parameters that are given as sharded
// inputs that can be changed to replicated sharding to improve performance.
class MpmdParameterReplication : public HloModulePass {
 public:
  // The `partition` object contains the mapping from partition color
  // to the assigned submesh. Only paramater smaller than (or equal)
  // in size to the `num_elements_cutoff` will be considered.
  explicit MpmdParameterReplication(HloPartition* partition,
                                    int64_t num_elments_cutoff)
      : partition_(partition), num_elements_cutoff_(num_elments_cutoff) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdParameterReplication() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-parameter-replication";
  }

 private:
  HloPartition* partition_;
  int64_t num_elements_cutoff_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_PARAMETER_REPLICATION_H_
