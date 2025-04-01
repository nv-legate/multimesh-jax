#include "xla/pjrt/legate/mpmd_hoist_shard_map_reduce.h"

#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/legate/mpmd_instruction.h"
#include "xla/pjrt/legate/mpmd_utils.h"
#include "xla/service/spmd/spmd_partitioner_util.h"

namespace xla {

bool operator==(const ReplicaGroup& lhs, const ReplicaGroup& rhs) {
  if (lhs.replica_ids_size() != rhs.replica_ids_size()) {
    return false;
  }

  for (int64_t index = 0; index < lhs.replica_ids_size(); ++index) {
    if (lhs.replica_ids(index) != rhs.replica_ids(index)) {
      return false;
    }
  }

  return true;
}

namespace {

struct OutputReduce {
  HloInstruction* reduce;
  // The MPMDShardShapeToFullShape instruction
  // that is the shard-mapped output of the reduce instruction
  HloInstruction* shard_map_output;
  // The root tuple operand index that this reduce
  // is summed into
  int64_t output_index;
};

// Given a vector of `output_reduces`, returns whether they
// all have the same replica groups and can therefore be
// merged into a single reduce.
bool AllReducesAreEquivalent(
    const absl::InlinedVector<OutputReduce, 2>& output_reduces) {
  if (output_reduces.size() == 1) {
    return true;
  }

  HloAllReduceInstruction* reference =
      static_cast<HloAllReduceInstruction*>(output_reduces.front().reduce);
  for (int index = 1; index < output_reduces.size(); ++index) {
    HloAllReduceInstruction* next =
        static_cast<HloAllReduceInstruction*>(output_reduces[index].reduce);
    if (next->replica_groups() != reference->replica_groups()) {
      return false;
    }
  }

  return true;
}

}  // namespace

absl::StatusOr<bool> MpmdHoistShardMapReduce::VisitLoop(
    HloInstruction* loop, const InstructionProperties& properties,
    HloPassCleanup& cleanup) {
  bool changed = false;
  HloComputation* computation = loop->called_computations()[0];
  HloInstruction* root = computation->root_instruction();

  absl::flat_hash_map</*tuple_index=*/int64_t,
                      absl::InlinedVector<OutputReduce, 2>>
      output_reduces;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->IsCustomCall("SPMDShardToFullShape")) {
      auto* operand = instruction->mutable_operand(0);
      if (operand->opcode() == HloOpcode::kAllReduce) {
        VLOG(5) << "Found shard-mapped all-reduce " << operand->name();
        const HloInstruction* additive_to_input{nullptr};
        const auto& instruction_properties = properties.Get(instruction);
        for (auto* input_summed_into :
             instruction_properties.additive_to_while_input) {
          VLOG(5) << operand->name() << " is summed into loop-carried add via "
                  << instruction->name();
          output_reduces[input_summed_into->tuple_index()].push_back(
              OutputReduce{.reduce = operand, .shard_map_output = instruction});
        }
      }
    }
  }

  // must be deleted in a deterministic order
  absl::flat_hash_set<HloInstruction*> to_delete;
  std::vector<HloInstruction*> delete_order;

  HloCloneContext context{loop->parent()->parent()};
  for (auto* user : loop->users()) {
    auto iter = output_reduces.find(user->tuple_index());
    if (iter != output_reduces.end()) {
      auto& reduces = iter->second;
      if (AllReducesAreEquivalent(reduces)) {
        changed = true;

        HloInstruction* reduce = reduces.front().reduce;

        // for perf experiments, may want to remove the all-reduces
        // instead of executing them to test perf upper bounds
        if (!remove_reduces_) {
          auto partitioned_shape =
              spmd::MakePartitionedShape(user->shape(), user->sharding());

          auto* manual_shard =
              loop->parent()->AddInstruction(HloInstruction::CreateCustomCall(
                  partitioned_shape, {user}, "SPMDFullToShardShape"));
          manual_shard->set_sharding(HloSharding::Manual());

          auto* operand = manual_shard;
          if (reduce->shape().element_type() != user->shape().element_type()) {
            operand =
                loop->parent()->AddInstruction(HloInstruction::CreateConvert(
                    ShapeUtil::ChangeElementType(
                        partitioned_shape, reduce->shape().element_type()),
                    manual_shard));
          }

          // TODO: handle reduce-scatter with shape changes
          auto* hoisted_reduce =
              loop->parent()->AddInstruction(reduce->CloneWithNewOperands(
                  ShapeUtil::ChangeElementType(partitioned_shape,
                                               reduce->shape().element_type()),
                  {operand}, &context));

          auto* output = hoisted_reduce;
          if (user->shape().element_type() !=
              hoisted_reduce->shape().element_type()) {
            output =
                loop->parent()->AddInstruction(HloInstruction::CreateConvert(
                    partitioned_shape, hoisted_reduce));
          }

          auto* full_shard =
              loop->parent()->AddInstruction(HloInstruction::CreateCustomCall(
                  user->shape(), {output}, "SPMDShardToFullShape"));
          full_shard->set_sharding(user->sharding_ptr());
          TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(full_shard));
          // this gets replaced, put it back
          TF_RETURN_IF_ERROR(manual_shard->ReplaceOperandWith(0, user));

          auto color = Color(reduce);
          if (color.has_value()) {
            AssignColor(manual_shard, *color);
            AssignColor(operand, *color);
            AssignColor(hoisted_reduce, *color);
            AssignColor(output, *color);
            AssignColor(full_shard, *color);
          }
        }

        for (OutputReduce& reduce : reduces) {
          VLOG(5) << reduce.reduce->name()
                  << " moving out of the loop, replacing with operand "
                  << reduce.reduce->operand(0)->name();
          TF_RETURN_IF_ERROR(reduce.reduce->ReplaceAllUsesWith(
              reduce.reduce->mutable_operand(0)));
          if (!to_delete.contains(reduce.reduce)) {
            delete_order.push_back(reduce.reduce);
            to_delete.insert(reduce.reduce);
          }
        }
      } else if (VLOG_IS_ON(5)) {
        for (auto&& output_reduce : reduces) {
          VLOG(5) << output_reduce.shard_map_output->name()
                  << " has different all-reduces: "
                  << output_reduce.reduce->ToString();
        }
      }
    }
  }
  for (auto* reduce : delete_order) {
    cleanup.RemoveInstruction(reduce);
  }
  return changed;
}

absl::StatusOr<bool> MpmdHoistShardMapReduce::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  auto properties = InstructionProperties::Create(module);
  HloPassCleanup cleanup;
  for (auto* instruction :
       module->entry_computation()->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kWhile) {
      TF_ASSIGN_OR_RETURN(bool instruction_changed,
                          VisitLoop(instruction, properties, cleanup));
      changed |= instruction_changed;
    }
  }

  TF_RETURN_IF_ERROR(cleanup.CleanUp());

  return changed;
}

}  // namespace xla