/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/critical_path_finder.h"

#include <limits>
#include <queue>

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/util.h"

namespace xla {

namespace {

absl::StatusOr<uint64_t> FirstCallDepth(
    HloInstruction* instruction, HloComputation* computation,
    const absl::flat_hash_map<const HloInstruction*, int64_t>& depth_map) {
  std::queue<HloInstruction*> queue{{instruction}};

  while (!queue.empty()) {
    HloInstruction* current = queue.front();
    queue.pop();

    if (current->opcode() == HloOpcode::kCall || current->IsRoot()) {
      return depth_map.at(current);
    }

    for (HloInstruction* user : current->users()) {
      queue.push(user);
    }
  }

  return absl::InvalidArgumentError("No call or root found");
}

}  // namespace

absl::StatusOr<uint64_t> CriticalRootTupleIndex(
    HloInstruction* call, HloComputation* computation,
    const absl::flat_hash_map<const HloInstruction*, int64_t>& depth_map) {
  uint64_t min_call_depth = std::numeric_limits<uint64_t>::max();
  uint64_t min_call_depth_root_id = 0;
  for (HloInstruction* user : call->users()) {
    // Expect all the users to be GTEs.
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return absl::InvalidArgumentError("Call has a user that is not a GTE: " +
                                        user->ToString());
    }
    uint64_t tuple_index = user->tuple_index();
    if (tuple_index >= computation->root_instruction()->operand_count()) {
      return absl::OutOfRangeError(
          absl::StrCat(user->name(), " has tuple index out of bounds"));
    }
    TF_ASSIGN_OR_RETURN(uint64_t call_depth,
                        FirstCallDepth(user, computation, depth_map));
    if (call_depth <= min_call_depth) {
      if (call_depth == min_call_depth &&
          tuple_index > min_call_depth_root_id) {
        continue;
      }
      min_call_depth = call_depth;
      min_call_depth_root_id = tuple_index;
    }
  }
  return min_call_depth_root_id;
}

}  // namespace xla