/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_LEGATE_MPMD_BUFFER_SCHEDULING_NAME_H_
#define XLA_PJRT_LEGATE_MPMD_BUFFER_SCHEDULING_NAME_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"

namespace xla {

// Assigns scheduling names to all instructions in the module
// as a means of assigning buffers to instructions
class MpmdAssignBufferSchedulingName : public HloModulePass {
 public:
  // The `partition` object containing the mapping from partition color
  // to the assigned submesh.
  explicit MpmdAssignBufferSchedulingName(HloPartition* partition)
      : partition_(partition) {}

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdAssignBufferSchedulingName() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-assign-buffer-scheduling-name";
  }

 private:
  HloPartition* partition_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_BUFFER_SCHEDULING_NAME_H_
