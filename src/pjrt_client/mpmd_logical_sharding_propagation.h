/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_LOGICAL_SHARDING_PROPAGATION_H_
#define XLA_PJRT_LEGATE_LOGICAL_SHARDING_PROPAGATION_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Propagates logical sharding annotations to operands/users
// and convert the logical sharding annotations into explicit device shardings
class MpmdLogicalShardingPropagation : public HloModulePass {
 public:
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdLogicalShardingPropagation() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-logical-sharding-propagation";
  }
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_LOGICAL_SHARDING_PROPAGATION_H_
