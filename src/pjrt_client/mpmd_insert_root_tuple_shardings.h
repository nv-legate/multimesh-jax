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

// If module has a sharded root tuple, propagate the sharding to it's operands
// and insert RootTupleRecolor custom calls to signify sharding overwrites
class MpmdInsertRootTupleShardings : public HloModulePass {
 public:
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
