#include "xla/pjrt/legate/mpmd_argument_recompute.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/pjrt/legate/legate_sharding.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"

namespace xla {

namespace {

// A fixed heuristic for when local buffers should be
// considered "large" and avoid recomputation
constexpr int64_t kLargeBufferSize = 1024 * 1024 * 8;

// A fixed heuristic for the ratio between a recomuptation
// flop and the cost to communicate a byte to a remote node.
constexpr int64_t kCrossMeshCommComputeRatio = 1000;

// A fixed heuristic for the ratio between a recomputation
// flop and the cost to communicate a byte from global memory.
constexpr int64_t kGlobalMemToComputeRatio = 100;

}  // namespace

MpmdArgumentRecompute::MpmdArgumentRecompute(
    HloPartition* partition, int64_t max_recompute_cost_allowed,
    std::optional<int64_t> global_mem_to_compute_ratio)
    : partition_(partition),
      max_recompute_cost_allowed_(max_recompute_cost_allowed),
      global_mem_to_compute_ratio_(
          global_mem_to_compute_ratio.value_or(kGlobalMemToComputeRatio)) {}

absl::StatusOr<bool> MpmdArgumentRecompute::MaybeRecomputeOperandFromArguments(
    const std::string& color, HloInstruction* instruction,
    HloInstruction* operand, HloComputation* computation,
    absl::flat_hash_map<const HloInstruction*, HloInstruction*>& clone_map,
    InstructionProperties& properties) {
  auto input_or_replicated_input_copy = [&](const HloInstruction* instruction) {
    return (instruction->opcode() == HloOpcode::kParameter &&
            !instruction->shape().IsTuple()) ||
           properties.ParameterAlias(instruction) ||
           (IsReplicatedOrNotSharded(instruction) &&
            properties.ParameterCopy(instruction));
  };

  const int64_t memory_cost =
      ShapeUtil::ByteSizeOf(GetSpmdShape(instruction), sizeof(void*)) *
      global_mem_to_compute_ratio_;

  const int64_t cross_mesh_cost = [&]() -> int64_t {
    const int64_t byte_size = ShapeUtil::ByteSizeOf(
        GetSpmdShape(instruction), /*pointer_size=*/sizeof(void*));
    if (byte_size > kLargeBufferSize) {
      auto devices = partition_->DevicesForColor(color);
      auto operand_color = Color(operand);
      if (operand_color.has_value()) {
        auto operand_devices = partition_->DevicesForColor(*operand_color);
        if (operand_devices != devices) {
          return byte_size * kCrossMeshCommComputeRatio;
        }
      }
    }
    return 0;
  }();

  const int64_t num_devices = partition_->NumDevicesForInstruction(instruction);

  const int64_t cost_cutoff = std::max(
      memory_cost, std::max(cross_mesh_cost, max_recompute_cost_allowed_));

  VLOG(5) << "visiting instruction " << instruction->name() << " and operand "
          << operand->name() << ":" << operand->shape()
          << " to check for recompute on " << color
          << ", cost_cutoff=" << cost_cutoff << ", memory_cost=" << memory_cost
          << ", max_recompute_cost_allowed=" << max_recompute_cost_allowed_
          << ", cross_mesh_cost=" << cross_mesh_cost;

  if (clone_map.contains(operand) || input_or_replicated_input_copy(operand)) {
    // someone else in our computation already cloned this or
    // it's a parameter or a parameter equivalent so no
    // extra cost to this instruction to use it and no changes are required
    return false;
  }

  std::vector<HloInstruction*> to_visit = {operand};
  std::vector<HloInstruction*> dfs_tree;

  absl::flat_hash_map<const HloInstruction*, /*done*/ bool> visiting;

  int64_t total_recompute_cost = 0;

  while (!to_visit.empty()) {
    HloInstruction* next = to_visit.back();

    auto iter = visiting.find(next);
    if (iter != visiting.end()) {
      // already visited
      to_visit.pop_back();
      if (iter->second) {
        VLOG(5) << "already visited " << next->name() << " in DFS for "
                << instruction->name();
        continue;
      }
      VLOG(5) << "finishing visit for " << next->name() << " in DFS for "
              << instruction->name();
      dfs_tree.push_back(next);
      iter->second = true;
      continue;
    }

    // if anything in the operand tree is sharded over a different number of
    // devices return false and quit
    // TODO: enable argument recompute across a tree with different meshes
    if (IsNontriviallySharded(next) && IsNontriviallySharded(instruction)) {
      if (next->sharding().tile_assignment().num_elements() !=
          instruction->sharding().tile_assignment().num_elements()) {
        VLOG(5) << "found misimatched sharding in operand tree for "
                << operand->name() << ":" << operand->shape()
                << " abandoning replication";
        return false;
      }
    }

    visiting[next] = false;  // visiting, not done
    if (next->IsCustomCall("SliceOffset")) {
      // consider the visit already done and don't push back
      // on the DFS tree
      visiting[next] = true;
      continue;
    }

    switch (next->opcode()) {
      case HloOpcode::kDot:
      case HloOpcode::kCustomCall:
        if (next->IsCustomCall("Reshard") || next->IsCustomCall("Sharding")) {
          auto* operand = next->mutable_operand(0);
          if (!clone_map.contains(operand) &&
              !input_or_replicated_input_copy(operand)) {
            to_visit.push_back(operand);
          }
          break;
        }
        VLOG(5) << "found potentially expensive operation " << next->name()
                << " in operand tree for " << operand->name() << ":"
                << operand->shape() << " abandoning replication";
        return false;
      case HloOpcode::kParameter:
        if (input_or_replicated_input_copy(next)) {
          break;
        }
        VLOG(5) << "found intermediate parameter " << next->name()
                << " in operand tree for " << operand->name() << ":"
                << operand->shape() << ", abandoning replication";
        // if here, then we must be hitting a parameter to a call/while
        // that is a temporary from another task, quit for now
        // only recompute within a computation
        return false;
      case HloOpcode::kCall:
      case HloOpcode::kWhile:
        return false;
      case HloOpcode::kIota:
      case HloOpcode::kConstant:
      case HloOpcode::kBroadcast:
      case HloOpcode::kReshape:
        // assume we can do this efficiently and it adds little extra cost
        break;
      case HloOpcode::kReduce:
        total_recompute_cost +=
            ShapeUtil::ElementsIn(GetSpmdShape(next->operand(0)));
        break;
      default:
        total_recompute_cost += ShapeUtil::ElementsIn(GetSpmdShape(next));
        break;
    }

    VLOG(5) << "total recompute cost for operand " << next->name() << " of "
            << instruction->name() << " is " << total_recompute_cost
            << " after visitng " << next->name() << ":" << next->shape() << "%"
            << next->sharding_or_default(HloSharding::Replicate());

    if (total_recompute_cost > cost_cutoff) {
      VLOG(5) << "total recompute cost " << total_recompute_cost << " for "
              << operand->name() << " " << operand->shape()
              << " exeeds threshold " << cost_cutoff
              << ", abandoning replication";
      return false;
    }

    for (int64_t index = 0; index < next->operand_count(); ++index) {
      auto* operand = next->mutable_operand(index);
      if (!clone_map.contains(operand) &&
          !input_or_replicated_input_copy(operand)) {
        to_visit.push_back(operand);
      }
    }
  }

  for (auto* to_clone : dfs_tree) {
    VLOG(5) << "visiting " << to_clone->name() << " in DFS operand tree of "
            << instruction->name();
    if (clone_map.contains(to_clone)) {
      // already cloned, instruction was visited multiple times in DFS tree
      continue;
    }
    absl::InlinedVector<HloInstruction*, 2> new_operands;
    for (auto* operand : to_clone->operands()) {
      auto operand_iter = clone_map.find(operand);
      // if this has been cloned, use the clone
      // if not, use the original instruction
      if (operand_iter == clone_map.end()) {
        new_operands.push_back(operand);
      } else {
        new_operands.push_back(operand_iter->second);
      }
    }
    VLOG(5) << "cloning " << to_clone->name()
            << " as part of recompute tree for operand " << operand->name()
            << " of " << instruction->name();
    auto* clone = computation->AddInstruction(
        to_clone->CloneWithNewOperands(to_clone->shape(), new_operands));

    if (!IsReplicatedOrNotSharded(clone) &&
        clone->sharding().tile_assignment().num_elements() != num_devices) {
      clone->clear_sharding();
    }

    clone_map[to_clone] = clone;
    properties.Clone(to_clone, clone);
    AssignColor(clone, color);
  }

  const int64_t operand_index = instruction->operand_index(operand);
  auto iter = clone_map.find(operand);
  if (iter == clone_map.end()) {
    return InvalidArgumentStrCat(operand->name(),
                                 " was not found in clone map for ",
                                 instruction->name());
  }

  TF_RETURN_IF_ERROR(
      instruction->ReplaceOperandWith(operand_index, iter->second));

  return true;
}

absl::StatusOr<bool> MpmdArgumentRecompute::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloComputation*> to_visit{module->entry_computation()};
  bool changed = false;
  auto properties = InstructionProperties::Create(module);

  HloPassCleanup cleanup;
  while (!to_visit.empty()) {
    HloComputation* computation = to_visit.back();
    to_visit.pop_back();

    bool computation_changed = false;
    absl::flat_hash_map<std::string, absl::flat_hash_map<const HloInstruction*,
                                                         HloInstruction*>>
        clones_per_color;
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kWhile) {
        to_visit.push_back(instruction->called_computations()[0]);
        continue;
      }

      if (instruction->opcode() == HloOpcode::kTuple ||
          properties.DerivedInput(instruction)) {
        continue;
      }

      auto color = Color(instruction);
      if (!color.has_value()) {
        continue;
      }

      for (int64_t index = 0; index < instruction->operand_count(); ++index) {
        auto* operand = instruction->mutable_operand(index);
        auto operand_color = Color(operand);
        if (operand_color.has_value() && *color != *operand_color &&
            !operand->IsCustomCall("SliceOffset")) {
          TF_ASSIGN_OR_RETURN(bool cloned_operand,
                              MaybeRecomputeOperandFromArguments(
                                  *color, instruction, operand, computation,
                                  clones_per_color[*color], properties));
          computation_changed |= cloned_operand;
          if (cloned_operand) {
            VLOG(5) << instruction->name() << " cloned operand tree for "
                    << operand->name() << ": " << operand->shape() << "%"
                    << operand->sharding_or_default(HloSharding::Replicate());
            if (instruction->IsCustomCall("Reshard")) {
              // we no longer need to mark a reshard here on the instruction, it
              // now has an entire operand tree with the same color beneath it
              TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(
                  instruction->mutable_operand(0)));

              cleanup.RemoveInstruction(instruction);
            }
          }
        }
      }
    }
    if (computation_changed) {
      cleanup.PruneComputation(computation);
    }
    changed |= computation_changed;
  }
  TF_RETURN_IF_ERROR(cleanup.CleanUp());
  return changed;
}

}  // namespace xla