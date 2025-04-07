/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_IOTA_SHARDING_SANITIZER_H_
#define XLA_PJRT_LEGATE_IOTA_SHARDING_SANITIZER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/util.h"

namespace xla {

class IotaShardingSanitizer : public HloModulePass {
 public:
  explicit IotaShardingSanitizer(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~IotaShardingSanitizer() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override { return "iota-sharding-sanitizer"; }

 private:
  HloPartition* partition_;

  absl::StatusOr<bool> Visit(HloComputation* comp);
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_IOTA_SHARDING_SANITIZER_H_
