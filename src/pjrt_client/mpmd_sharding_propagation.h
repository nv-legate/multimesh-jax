/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PJRT_LEGATE_MPMD_SHARDING_PROPAGATION_H_
#define XLA_PJRT_LEGATE_MPMD_SHARDING_PROPAGATION_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

struct MpmdShardingPropagationConfig {
  bool shard_iota{false};
  std::optional<int64_t> min_shard_override_size;
};

// Propagates sharding in an MPMD-compatible way. The pass assumes
// that tasks assigned to different partition colors have been
// grouped into distinct computations. This applies sharding
// propagation to each computation separately and therefore allows
// sharding propagation over different numbers of devices.
class MpmdShardingPropagation : public HloModulePass {
 public:
  enum class PropagationMode {
    ForwardInputOutput,
    BackwardInputOutput,
    ForwardFull
  };

  // The `partition` object contains the mapping from partition color
  // to the assigned submesh. The `mode` determines whether sharding
  // should proceed forward or backward and whether sharding should
  // be assigned to all instructions or only the parameter/root instructions.
  // The `cfg` sets optional configuration values.
  explicit MpmdShardingPropagation(HloPartition* partition,
                                   PropagationMode mode,
                                   MpmdShardingPropagationConfig cfg = {});

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  ~MpmdShardingPropagation() override = default;

  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;

  absl::string_view name() const override {
    return "mpmd-sharding-propagation";
  }

 private:
  // Propagate any derived shardings from the parameters of the `computation`
  // to their inputs to the call instruction.
  absl::Status ShardBackward(HloComputation* computation,
                             const InstructionProperties& properties,
                             HloPassCleanup& cleanup);

  // Propagate any derived shardings from the root of the `computation`
  // to users of the called computation.
  absl::Status ShardForward(HloComputation* computation,
                            InstructionProperties& properties,
                            HloPassCleanup& cleanup);

  // If a call `instruction` is encountered, propagate shardings
  // from the operands to the call into the parameters of the
  // called computation and propagate within the computation.
  absl::Status ShardForwardCallInstruction(HloInstruction* instruction,
                                           InstructionProperties& properties,
                                           HloPassCleanup& cleanup);

  // If a while `instruction` is encountered, propagate shardings
  // from the tuple input operands into the while loop
  // and propagate shardings across while loop called computations.
  absl::Status ShardForwardWhileInstruction(HloInstruction* instruction,
                                            InstructionProperties& properties,
                                            HloPassCleanup& cleanup);

  // Replaces the given `call` with a call to the `sharded_computation`
  // that has shardings assigned to all inputs, intermediates, and outputs.
  absl::Status ReplaceCallWithShardedTwin(HloInstruction* call,
                                          HloComputation* sharded_computation,
                                          HloModule* module,
                                          InstructionProperties& properties,
                                          HloPassCleanup& cleanup);

  absl::StatusOr<std::unique_ptr<HloModule>> PropagateComputation(
      HloComputation* computation, HloInstruction* call,
      const InstructionProperties& properties, bool propagate_to_parameters);

  void PropagateParamShardingsBackward(HloComputation* sharded_computation,
                                       HloInstruction* call_instruction,
                                       const InstructionProperties& properties);

  void PropagateOutputShardingsForward(HloComputation* sharded_computation,
                                       HloInstruction* call_instruction,
                                       const InstructionProperties& properties);

  void FixReshapeSharding(HloInstruction* call);

  HloPartition* partition_;
  bool shard_iota_;
  PropagationMode mode_;
  int64_t min_shard_override_size_;
};

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_SHARDING_PROPAGATION_H_
