/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_utils.h"

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {
namespace {

static absl::flat_hash_set<std::string> kNoOpCustomCalls = {
    "Reshard",
    "Sharding",
    "SPMDFullToShardShape",
    "SPMDShardToFullShape",
};

}  // namespace

absl::Status HloPassCleanup::CleanUp() {
  for (auto* instruction : instructions_to_remove_) {
    TF_RETURN_IF_ERROR(instruction->parent()->RemoveInstruction(instruction));
  }
  for (auto* computation : computations_to_prune_) {
    TF_RETURN_IF_ERROR(RemoveUnusedInstructions(computation));
  }
  for (auto* computation : computations_to_remove_) {
    TF_RETURN_IF_ERROR(
        computation->parent()->RemoveEmbeddedComputation(computation));
  }
  return absl::OkStatus();
}

bool GetEnvOption(absl::string_view name, bool deflt) {
  const char* env = getenv(name.data());
  if (env) {
    return bool(std::atoi(env));
  }
  return deflt;
}

bool IsReplicatedOrNotSharded(const HloInstruction* instruction) {
  return !instruction->has_sharding() || instruction->sharding().IsReplicated();
}

bool IsNontriviallySharded(const HloInstruction* instruction) {
  return instruction->has_sharding() && instruction->sharding().IsReplicated();
}

HloInstruction* GetTupleOrComputationAlias(HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kGetTupleElement) {
    auto* operand = instruction->mutable_operand(0);
    switch (operand->opcode()) {
      case HloOpcode::kTuple:
        return operand->mutable_operand(instruction->tuple_index());
      case HloOpcode::kParameter:
        return nullptr;
      case HloOpcode::kCall:
      case HloOpcode::kWhile:
        return operand->called_computations()[0]
            ->root_instruction()
            ->mutable_operand(instruction->tuple_index());
      case HloOpcode::kOptimizationBarrier:
        return operand->mutable_operand(0)->mutable_operand(
            instruction->tuple_index());
      default:
        return nullptr;
    }
  }

  switch (instruction->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kWhile:
      return instruction->called_computations()[0]->root_instruction();
    default:
      return nullptr;
  }
}

HloInstruction* UnwrapCustomCall(HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kCustomCall) {
    return UnwrapCustomCall(instruction);
  }
  return instruction;
}

HloInstruction* GetComputationRootTuplePartner(HloInstruction* gte,
                                               HloComputation* comp) {
  return comp->root_instruction()->mutable_operand(gte->tuple_index());
}

HloInstruction* GetTupleElement(HloInstruction* tuple, int64_t index) {
  for (auto* user : tuple->users()) {
    if (user->tuple_index() == index) {
      return user;
    }
  }
  return nullptr;
}

// Helper function for computing the product of all dimension
// in the shape of the instruction. This is not the size in
// bytes of the instruction, only the number of elements.
int64_t DimensionProduct(const HloInstructionProto& instr) {
  int64_t prod = 1;
  for (auto dim : instr.shape().dimensions()) {
    prod *= dim;
  }
  return prod;
}

absl::StatusOr<zuku::DeviceList> CreateDeviceList(
    const std::vector<int64_t>& devices) {
  for (size_t idx = 1; idx < devices.size(); ++idx) {
    if (devices[idx] < devices[idx - 1]) {
      return Unimplemented("non-contiguous device lists not yet supported");
    }
  }
  return zuku::DeviceList::Create(devices.front(), devices.size());
}

absl::StatusOr<zuku::DeviceList> CreateDeviceList(
    const TileAssignment& tile_assignment) {
  if (tile_assignment.iota().has_value()) {
    const int64_t start = tile_assignment.iota()->offset();
    const int64_t stop = start + tile_assignment.iota()->num_elements();
    return zuku::DeviceList{
        {.start = tile_assignment.iota()->offset(),
         .num_devices = tile_assignment.iota()->num_elements()}};
  }

  const int64_t first_device = tile_assignment.first();
  int64_t prev_device = first_device - 1;
  for (auto&& dev : tile_assignment.array()) {
    if (dev != (prev_device + 1)) {
      return Unimplemented(
          "non-iota tile assignments not yet supported for DeviceList");
    }
    prev_device = dev;
  }
  const int64_t last_device = prev_device + 1;
  return zuku::DeviceList{
      {.start = first_device, .num_devices = last_device - first_device}};
}

void InstructionProperties::Entry::CopyFromAlias(
    const InstructionProperties::Entry& entry) {
  derived_constant = entry.derived_constant;
  derived_parameter = entry.derived_parameter;
  allow_recomputation = entry.allow_recomputation;
  parameter_number = entry.parameter_number;
  loop_carried_index = entry.loop_carried_index;
  loop_parameter_index = entry.loop_parameter_index;
  elementwise_connected_to_entry_output =
      entry.elementwise_connected_to_entry_output;
  equivalent_to_while_output = entry.equivalent_to_while_output;
  additive_to_while_input = entry.additive_to_while_input;
  loop_carried_initializer = entry.loop_carried_initializer;
  parameter_copy = entry.parameter_copy;
}

void InstructionProperties::ForEachBackwardAlias(HloInstruction* instruction,
                                                 AliasVisitFn fn) const {
  HloInstruction* next = instruction;
  auto iter = entries_.find(instruction);
  CHECK(iter != entries_.end());
  while (iter->second.backward_alias) {
    fn(iter->second.backward_alias);
    next = iter->second.backward_alias;
    iter = entries_.find(next);
    CHECK(iter != entries_.end());
  }
}

void InstructionProperties::ForEachForwardAlias(HloInstruction* instruction,
                                                AliasVisitFn fn) const {
  HloInstruction* next = instruction;
  auto iter = entries_.find(instruction);
  CHECK(iter != entries_.end());
  while (iter->second.forward_alias) {
    fn(iter->second.forward_alias);
    next = iter->second.forward_alias;
    iter = entries_.find(next);
    CHECK(iter != entries_.end());
  }
}

void InstructionProperties::ForEachAlias(HloInstruction* instruction,
                                         AliasVisitFn fn) const {
  ForEachBackwardAlias(instruction, fn);
  ForEachForwardAlias(instruction, fn);
}

void InstructionProperties::ForEachAliasAndSelf(HloInstruction* instruction,
                                                AliasVisitFn fn) const {
  fn(instruction);
  ForEachBackwardAlias(instruction, fn);
  ForEachForwardAlias(instruction, fn);
}

void InstructionProperties::ForEachBackwardAliasAndSelf(
    HloInstruction* instruction, AliasVisitFn fn) const {
  fn(instruction);
  ForEachBackwardAlias(instruction, fn);
}

void InstructionProperties::ForEachForwardAliasAndSelf(
    HloInstruction* instruction, AliasVisitFn fn) const {
  fn(instruction);
  ForEachForwardAlias(instruction, fn);
}

bool InstructionProperties::AllowRecomputation(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties";
  }
  return iter->second.allow_recomputation;
}

const InstructionProperties::Entry& InstructionProperties::Get(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties";
  }
  return iter->second;
}

HloInstruction* InstructionProperties::ForwardAlias(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties";
  }
  return iter->second.forward_alias;
}

HloInstruction* InstructionProperties::BackwardAlias(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties";
  }
  return iter->second.backward_alias;
}

void InstructionProperties::AddBackwardAlias(HloInstruction* instruction,
                                             Entry& instruction_props,
                                             HloInstruction* alias,
                                             Entry& alias_props) {
  instruction_props.backward_alias = alias;
}

void InstructionProperties::AddBackwardAlias(HloInstruction* instruction,
                                             HloInstruction* alias) {
  AddBackwardAlias(instruction, entries_[instruction], alias, entries_[alias]);
}

void InstructionProperties::AddForwardAlias(HloInstruction* instruction,
                                            Entry& instruction_props,
                                            HloInstruction* alias,
                                            Entry& alias_props) {
  instruction_props.forward_alias = alias;
}

void InstructionProperties::AddForwardAlias(HloInstruction* instruction,
                                            HloInstruction* alias) {
  AddForwardAlias(instruction, entries_[instruction], alias, entries_[alias]);
}

void InstructionProperties::AddDefaultProperties(
    const HloInstruction* instruction) {
  entries_[instruction];
}

HloInstruction* DealiasLoopInput(HloInstruction* instruction) {
  auto* operand = instruction;
  while (operand) {
    if (operand->IsCustomCall("Reshard")) {
      operand = operand->mutable_operand(0);
    } else if (operand->opcode() == HloOpcode::kGetTupleElement &&
               operand->operand(0)->opcode() == HloOpcode::kParameter) {
      return operand;
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

void InstructionProperties::AddBackwardProperties(HloComputation* computation,
                                                  bool loop) {
  HloInstruction* loop_arg_tuple =
      loop ? computation->parameter_instruction(0) : nullptr;

  auto propagate_elementwise = [](const InstructionProperties::Entry& src,
                                  InstructionProperties::Entry& target) {
    if (!target.elementwise_connected_to_entry_output) {
      target.elementwise_connected_to_entry_output =
          src.elementwise_connected_to_entry_output;
    }
  };

  auto propagate_additive_elementwise =
      [&propagate_elementwise](const InstructionProperties::Entry& src,
                               InstructionProperties::Entry& target) {
        propagate_elementwise(src, target);
        target.additive_to_while_input = src.additive_to_while_input;
      };

  auto propagate_no_op_elementwise =
      [&propagate_elementwise](const InstructionProperties::Entry& src,
                               InstructionProperties::Entry& target) {
        propagate_elementwise(src, target);
        target.equivalent_to_while_output = src.equivalent_to_while_output;
        target.additive_to_while_input = src.additive_to_while_input;
      };

  auto postorder = computation->MakeInstructionPostOrder();
  for (auto iter = postorder.rbegin(); iter != postorder.rend(); ++iter) {
    HloInstruction* instruction = *iter;
    auto& properties = entries_[instruction];
    HloInstruction* additive_to_root = nullptr;
    HloInstruction* elementwise_connected_output = nullptr;
    switch (instruction->opcode()) {
      case HloOpcode::kAdd: {
        auto* lhs = instruction->mutable_operand(0);
        auto& lhs_properties = entries_[lhs];
        propagate_elementwise(properties, lhs_properties);
        auto* lhs_loop_input = DealiasLoopInput(lhs);

        auto* rhs = instruction->mutable_operand(1);
        auto& rhs_properties = entries_[rhs];
        propagate_elementwise(properties, rhs_properties);
        auto* rhs_loop_input = DealiasLoopInput(rhs);

        if (lhs_loop_input) {
          properties.additive_to_while_input.push_back(lhs_loop_input);
          rhs_properties.additive_to_while_input.push_back(lhs_loop_input);
        } else if (rhs_loop_input) {
          properties.additive_to_while_input.push_back(rhs_loop_input);
          lhs_properties.additive_to_while_input.push_back(rhs_loop_input);
        } else if (!properties.additive_to_while_input.empty()) {
          // TODO: support operands with multiple users
          if (lhs->users().size() == 1) {
            propagate_additive_elementwise(properties, lhs_properties);
          }
          if (rhs->users().size() == 1) {
            propagate_additive_elementwise(properties, rhs_properties);
          }
        }
        break;
      }
      case HloOpcode::kCustomCall:
        if (kNoOpCustomCalls.contains(instruction->custom_call_target())) {
          propagate_no_op_elementwise(properties,
                                      entries_[instruction->operand(0)]);
        }
        break;
      case HloOpcode::kReduce:
        // if a sum reduce
        if (instruction->called_computations()[0]
                ->root_instruction()
                ->opcode() == HloOpcode::kAdd) {
          // this is
          entries_[instruction->operand(0)].additive_to_while_input =
              properties.additive_to_while_input;
        }
        break;
      case HloOpcode::kTranspose:
      case HloOpcode::kReshape:
      case HloOpcode::kConvert:
      case HloOpcode::kBitcast: {
        propagate_no_op_elementwise(properties,
                                    entries_[instruction->operand(0)]);
        break;
      }
      // binary elementwise
      case HloOpcode::kMultiply:
      case HloOpcode::kDivide:
      case HloOpcode::kOr:
      case HloOpcode::kSubtract:
      case HloOpcode::kSelect:
      case HloOpcode::kAnd: {
        propagate_elementwise(properties, entries_[instruction->operand(0)]);
        propagate_elementwise(properties, entries_[instruction->operand(1)]);
        break;
      }
      // unary elementwise
      case HloOpcode::kSin:
      case HloOpcode::kCos:
      case HloOpcode::kExp:
      case HloOpcode::kCeil:
      case HloOpcode::kCopy:
      case HloOpcode::kLog:
      case HloOpcode::kNegate:
      case HloOpcode::kNot:
      case HloOpcode::kReducePrecision:
      case HloOpcode::kSqrt:
      case HloOpcode::kTan:
      case HloOpcode::kTanh: {
        propagate_elementwise(properties, entries_[instruction->operand(0)]);
        break;
      }
      case HloOpcode::kCall: {
        auto* subcomp = instruction->called_computations()[0];
        if (instruction->shape().IsTuple()) {
          auto* root = subcomp->root_instruction();
          for (auto* user : instruction->users()) {
            auto* root_input = root->mutable_operand(user->tuple_index());
            propagate_elementwise(entries_[user], entries_[root_input]);
          }
        }
        AddBackwardProperties(subcomp, /*loop=*/false);
        for (int64_t index = 0; index < instruction->operand_count(); ++index) {
          auto* operand = instruction->operand(index);
          auto* param = subcomp->parameter_instruction(index);
          propagate_elementwise(entries_[param], entries_[operand]);
        }
        break;
      }

      case HloOpcode::kWhile: {
        auto* subcomp = instruction->called_computations()[0];
        if (instruction->opcode() == HloOpcode::kWhile) {
          if (subcomp->root_instruction()->opcode() == HloOpcode::kTuple) {
            for (auto* operand : subcomp->root_instruction()->operands()) {
              entries_[operand].equivalent_to_while_output = operand;
            }
          } else {
            entries_[subcomp->root_instruction()].equivalent_to_while_output =
                computation->root_instruction();
          }
        }
        auto* root = subcomp->root_instruction();
        for (auto* user : instruction->users()) {
          auto* root_input = root->mutable_operand(user->tuple_index());
          propagate_elementwise(entries_[user], entries_[root_input]);
        }
        AddBackwardProperties(subcomp, /*loop=*/true);
        auto* input_tuple = instruction->operand(0);
        for (auto* user : subcomp->parameter_instruction(0)->users()) {
          auto* tuple_operand = input_tuple->operand(user->tuple_index());
          propagate_elementwise(entries_[user], entries_[tuple_operand]);
        }
        break;
      }
      default:
        break;
    }
  }
}

void InstructionProperties::AddForwardProperties(HloComputation* computation) {
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    auto& prop = entries_[instruction];
    // aliases may have been written prior to recursing into this computation
    switch (instruction->opcode()) {
      case HloOpcode::kConstant:
        prop.derived_constant = true;
        break;
      case HloOpcode::kTranspose:
      case HloOpcode::kBroadcast:
      case HloOpcode::kReshape:
      case HloOpcode::kConvert: {
        const auto& operand_prop = entries_[instruction->operand(0)];
        if (operand_prop.derived_constant || operand_prop.derived_parameter) {
          VLOG(5) << instruction->name() << " is a derived paramter/constant";
        }
        prop.derived_constant = operand_prop.derived_constant;
        prop.derived_parameter = operand_prop.derived_parameter;
        break;
      }
      case HloOpcode::kCall: {
        auto* comp = instruction->called_computations()[0];
        for (int64_t parameter_number = 0;
             parameter_number < comp->num_parameters(); ++parameter_number) {
          auto* inner_param = comp->parameter_instruction(parameter_number);
          auto* input_alias = instruction->mutable_operand(parameter_number);
          auto& param_properties = entries_[input_alias];
          Entry inner_properties;
          inner_properties.CopyFromAlias(param_properties);
          AddBackwardAlias(inner_param, inner_properties, input_alias,
                           param_properties);
          entries_[inner_param] = std::move(inner_properties);
        }
        AddForwardProperties(comp);
        if (instruction->shape().IsTuple()) {
          auto* root = comp->root_instruction();
          for (auto* user : instruction->users()) {
            auto* matching_root = root->mutable_operand(user->tuple_index());
            auto& user_properties = entries_[user];
            auto& root_properties = entries_.at(matching_root);
            AddForwardAlias(matching_root, root_properties, user,
                            user_properties);
            AddBackwardAlias(user, user_properties, matching_root,
                             root_properties);
            entries_[user] = std::move(user_properties);
          }
        } else {
          AddForwardAlias(comp->root_instruction(), instruction);
        }
        break;
      }
      case HloOpcode::kWhile: {
        auto* input_tuple = instruction->mutable_operand(0);
        auto* comp = instruction->called_computations()[0];
        auto* root_tuple = comp->root_instruction();
        CHECK(root_tuple->opcode() == HloOpcode::kTuple);
        CHECK(comp->parameter_instructions().size() == 1);
        auto* body_tuple_param = comp->parameter_instruction(0);
        CHECK(body_tuple_param->shape().IsTuple());
        for (auto* user : body_tuple_param->users()) {
          const int64_t parameter_number = user->tuple_index();
          auto* input_alias = input_tuple->mutable_operand(parameter_number);
          auto& user_properties = entries_[user];
          auto& input_properties = entries_.at(input_alias);
          user_properties.CopyFromAlias(input_properties);
          AddBackwardAlias(user, user_properties, input_alias,
                           input_properties);
          VLOG(5) << user->name() << " has backwards alias "
                  << input_alias->name();
          auto* matching_root_input =
              root_tuple->mutable_operand(parameter_number);

          if (matching_root_input == user) {
            // this is a carried-through parameter
            user_properties.loop_parameter_index = parameter_number;
            ForEachBackwardAlias(user, [&](HloInstruction* i) {
              entries_[i].loop_parameter_index = parameter_number;
            });
          } else {
            user_properties.loop_carried_index = parameter_number;
            user_properties.loop_carried_initializer = input_alias;
            ForEachBackwardAliasAndSelf(input_alias, [&](HloInstruction* i) {
              entries_[i].loop_parameter_index = parameter_number;
            });

            input_properties.loop_carried_index = parameter_number;
            input_properties.loop_carried_initializer = input_alias;

            auto& root_properties = entries_[matching_root_input];
            root_properties.loop_carried_index = parameter_number;
            root_properties.loop_carried_initializer = input_alias;
          }
        }

        for (auto* user : instruction->users()) {
          auto* matching_root_input =
              root_tuple->mutable_operand(user->tuple_index());
          auto& input_properties = entries_[matching_root_input];
          Entry user_properties;
          user_properties.CopyFromAlias(input_properties);
          AddBackwardAlias(user, user_properties, matching_root_input,
                           input_properties);
          AddForwardAlias(matching_root_input, input_properties, user,
                          user_properties);
          entries_[user] = std::move(user_properties);
        }
        AddForwardProperties(comp);
        break;
      }
      case HloOpcode::kOptimizationBarrier: {
        if (instruction->shape().IsTuple()) {
          auto* input_tuple = instruction->mutable_operand(0);
          for (auto* user : instruction->users()) {
            auto* alias = input_tuple->mutable_operand(user->tuple_index());
            AddBackwardAlias(user, alias);
          }
        }
        break;
      }
      case HloOpcode::kCopy: {
        // if something is a copy at this point, it was inserted to force
        // a new physical version of something
        prop.allow_recomputation = false;
        auto& copy_props = entries_[instruction];
        const auto& operand_props = entries_[instruction->operand(0)];
        if (operand_props.parameter_number.has_value() ||
            operand_props.parameter_copy) {
          copy_props.parameter_copy = true;
        }
        break;
      }
      case HloOpcode::kCustomCall: {
        if (instruction->IsCustomCall("Reshard")) {
          auto& reshard_props = entries_[instruction];
          const auto& operand_props = entries_[instruction->operand(0)];
          reshard_props.derived_parameter = operand_props.derived_parameter;
          if (operand_props.parameter_number.has_value()) {
            reshard_props.parameter_copy = true;
          }
        }
      }
      default:
        break;
    }
  }
}

void InstructionProperties::AddEntryProperties(HloComputation* computation) {
  int64_t parameter_number = 0;
  for (auto* instruction : computation->parameter_instructions()) {
    auto& props = entries_[instruction];
    props.derived_parameter = true;
    props.parameter_number = parameter_number++;
    entries_[instruction].derived_parameter = true;
  }

  if (computation->root_instruction()->opcode() == HloOpcode::kTuple) {
    for (auto* operand : computation->root_instruction()->operands()) {
      entries_[operand].elementwise_connected_to_entry_output = operand;
    }
  } else {
    entries_[computation->root_instruction()]
        .elementwise_connected_to_entry_output =
        computation->root_instruction();
  }
}

void InstructionProperties::Add(const InstructionProperties& other) {
  for (const auto& [instruction, entry] : other.entries_) {
    entries_[instruction].CopyFromAlias(entry);
  }
}

InstructionProperties InstructionProperties::Create(
    HloComputation* computation) {
  InstructionProperties properties;
  properties.AddEntryProperties(computation);

  auto alias_string = [&](HloInstruction* i) -> std::string {
    if (i == nullptr) {
      return "null";
    }
    return std::string(i->name());
  };

  properties.AddForwardProperties(computation);
  properties.AddBackwardProperties(computation, /*loop=*/false);

  if (VLOG_IS_ON(5)) {
    auto print_or_null = [](const HloInstruction* i) {
      if (i) {
        return std::string(i->name());
      }
      return std::string("null");
    };

    auto print_vector = [&print_or_null](auto& vec) {
      if (vec.empty()) {
        return std::string("{}");
      }
      std::string str = "{ ";
      for (auto* i : vec) {
        absl::StrAppend(&str, print_or_null(i), ", ");
      }
      absl::StrAppend(&str, "}");
      return str;
    };
    for (const auto& [instruction, props] : properties.map()) {
      VLOG(5) << instruction->name() << ","
              << "derived_constant=" << std::boolalpha << props.derived_constant
              << ",derived_parameter=" << props.derived_parameter
              << ",loop_carried_index=" << props.loop_carried_index.value_or(-1)
              << ",backward_alias=" << alias_string(props.backward_alias)
              << ",forward_alias=" << alias_string(props.forward_alias)
              << ",loop_carried_initializer="
              << alias_string(props.loop_carried_initializer)
              << ",parameter_copy=" << props.parameter_copy
              << ",elementwise_connected_to_entry_output="
              << print_or_null(props.elementwise_connected_to_entry_output)
              << ",equivalent_to_while_output="
              << print_or_null(props.equivalent_to_while_output)
              << ",additive_to_while_input="
              << print_vector(props.additive_to_while_input);
    }
  }
  return properties;
}

InstructionProperties InstructionProperties::Create(HloModule* module) {
  return Create(module->entry_computation());
}

void InstructionProperties::Clone(const HloInstruction* source,
                                  const HloInstruction* target) {
  Entry entry;
  entry.CopyFromAlias(entries_[source]);
  entries_[target] = std::move(entry);
}

std::optional<int64_t> InstructionProperties::ParameterNumber(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties assigned";
  }
  return iter->second.parameter_number;
}

bool InstructionProperties::ParameterAlias(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties assigned";
  }
  return iter->second.parameter_number.has_value();
}

bool InstructionProperties::ParameterCopy(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties assigned";
  }
  return iter->second.parameter_copy;
}

bool InstructionProperties::DerivedConstant(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties assigned";
  }
  return iter->second.derived_constant;
}

bool InstructionProperties::DerivedParameter(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties assigned";
  }
  return iter->second.derived_parameter;
}

bool InstructionProperties::DerivedInput(
    const HloInstruction* instruction) const {
  auto iter = entries_.find(instruction);
  if (iter == entries_.end()) {
    LOG(FATAL) << instruction->name() << " has no properties assigned";
  }
  return iter->second.derived_parameter || iter->second.derived_constant;
}

absl::Status RemoveUnusedInstructions(HloComputation* computation) {
  // there may be dead instructions here that prevented cleanup
  // during comp->RemoveInstructionAndUnusedOperands
  // loop backwards and remove unused instructions
  std::vector<HloInstruction*> final_post_order =
      computation->MakeInstructionPostOrder();
  // skip the first (root) instruction
  for (auto iter = final_post_order.rbegin(); iter != final_post_order.rend();
       ++iter) {
    HloInstruction* instruction = *iter;
    if (instruction != computation->root_instruction() &&
        instruction->users().empty() &&
        instruction->opcode() != HloOpcode::kParameter) {
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
    }
  }
  return absl::OkStatus();
}

absl::Status RemoveInstructionBackToParameters(HloComputation* computation,
                                               HloInstruction* instruction) {
  absl::InlinedVector<HloInstruction*, 2> delete_tree{instruction};
  absl::flat_hash_set<HloInstruction*> deleted;
  while (!delete_tree.empty()) {
    auto* next = delete_tree.back();
    delete_tree.pop_back();
    if (deleted.contains(next)) {
      continue;
    }
    if (next->users().empty()) {
      if (next->opcode() != HloOpcode::kParameter) {
        for (auto* operand : next->mutable_operands()) {
          delete_tree.push_back(operand);
        }
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(next));
        deleted.insert(next);
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace xla
