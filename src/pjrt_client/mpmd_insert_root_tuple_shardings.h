/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_MPMD_INSERT_ROOT_TUPLE_SHARDINGS_H_
#define XLA_PJRT_LEGATE_MPMD_INSERT_ROOT_TUPLE_SHARDINGS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

// Assigns a coloring to all non-trivial operations in the HLO module
// to create a partition of all operations based on user annotations.
// Partitions (colors) are assigned to instructions as a frontend attribute.
class MpmdInsertRootTupleShardings : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdInsertRootTupleShardings(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdInsertRootTupleShardings() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-insert-root-tuple-shardings";
  }

 private:
  //
  absl::StatusOr<bool> HandleRootTupleShardings(HloModule* module);

  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_INSERT_ROOT_TUPLE_SHARDINGS_H_
