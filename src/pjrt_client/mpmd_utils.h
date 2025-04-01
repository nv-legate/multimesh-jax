#ifndef XLA_PJRT_LEGATE_MPMD_UTILS_H_
#define XLA_PJRT_LEGATE_MPMD_UTILS_H_

#include "absl/status/status.h"
#include "src/zuku/mesh.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

// A clean that enables deferring instruction removal and cleanup
// until the end of the HLO pass.
//
// The InstructionProperties class assumes the module is immutable
// or (at least) that instructions are not removed. Removing
// instructions may invalidate fields in the InstructionProperties map
// and lead to accessing freed pointers.
class HloPassCleanup {
 public:
  void PruneComputation(HloComputation* computation) {
    computations_to_prune_.push_back(computation);
  }

  void RemoveInstruction(HloInstruction* instruction) {
    instructions_to_remove_.push_back(instruction);
  }

  void RemoveComputation(HloComputation* computation) {
    computations_to_remove_.push_back(computation);
  }

  absl::Status CleanUp();

 private:
  std::vector<HloComputation*> computations_to_remove_;
  std::vector<HloComputation*> computations_to_prune_;
  std::vector<HloInstruction*> instructions_to_remove_;
};

bool GetEnvOption(absl::string_view name, bool deflt);
HloInstruction* GetTupleElement(HloInstruction* tuple, int64_t index);
HloInstruction* GetTupleOrComputationAlias(HloInstruction* instruction);
HloInstruction* GetComputationRootTuplePartner(HloInstruction* gte,
                                               HloComputation* comp);

bool IsReplicatedOrNotSharded(const HloInstruction* instruction);

bool IsNontriviallySharded(const HloInstruction* instruction);

absl::StatusOr<zuku::DeviceList> CreateDeviceList(
    const std::vector<int64_t>& devices);

absl::StatusOr<zuku::DeviceList> CreateDeviceList(
    const TileAssignment& tile_assignment);

int64_t DimensionProduct(const HloInstructionProto& instr);

HloInstruction* UnwrapCustomCall(HloInstruction* instruction);

class InstructionProperties {
 public:
  struct Entry {
    bool derived_constant{false};
    bool derived_parameter{false};
    bool allow_recomputation{true};
    bool parameter_copy{false};
    HloInstruction* forward_alias{nullptr};
    HloInstruction* backward_alias{nullptr};
    std::optional<int64_t> parameter_number{std::nullopt};
    std::optional<int64_t> loop_carried_index{std::nullopt};
    std::optional<int64_t> loop_parameter_index{std::nullopt};
    HloInstruction* elementwise_connected_to_entry_output{nullptr};
    // is connected to the root by simple operations that don't
    // change the logical values such as convert, reshape, transpose
    HloInstruction* equivalent_to_while_output{nullptr};
    // part of a tree that is a loop-carried sum.
    // if there is some operation C = A + B where A is a loop input
    // and C is `equivalent_to_while_output`, then this is part of the
    // operand tree of C where the only operations are add or reshape/convert
    // or part of the user tree between C and the root output
    absl::InlinedVector<const HloInstruction*, 2> additive_to_while_input;
    HloInstruction* loop_carried_initializer{nullptr};

    Entry(Entry&&) = default;
    Entry& operator=(Entry&&) = default;
    Entry() = default;

   private:
    Entry(const Entry&) = delete;

    friend class InstructionProperties;
    void CopyFromAlias(const Entry& entry);
  };

  using AliasVisitFn = absl::FunctionRef<void(HloInstruction*)>;

  using ColorVisitFn =
      absl::FunctionRef<void(const HloInstruction*, const std::string&)>;

  const auto& map() const { return entries_; }

  const auto& Operands(const HloInstruction* instruction) const {
    const HloInstruction* backward_self = instruction;
    while (auto* alias = BackwardAlias(backward_self)) {
      backward_self = alias;
    }
    return backward_self->operands();
  }

  const auto& Users(const HloInstruction* instruction) const {
    const HloInstruction* forward_self = instruction;
    while (auto* alias = ForwardAlias(forward_self)) {
      forward_self = alias;
    }
    return forward_self->users();
  }

  void ForEachBackwardAlias(HloInstruction* instruction, AliasVisitFn fn) const;

  void ForEachForwardAlias(HloInstruction* instruction, AliasVisitFn fn) const;

  void ForEachAlias(HloInstruction* instruction, AliasVisitFn fn) const;

  void ForEachBackwardAliasAndSelf(HloInstruction* instruction,
                                   AliasVisitFn fn) const;

  void ForEachForwardAliasAndSelf(HloInstruction* instruction,
                                  AliasVisitFn fn) const;

  void ForEachAliasAndSelf(HloInstruction* instruction, AliasVisitFn fn) const;

  bool DerivedConstant(const HloInstruction* instruction) const;

  bool ParameterCopy(const HloInstruction* instruction) const;

  bool ParameterAlias(const HloInstruction* instruction) const;

  bool DerivedParameter(const HloInstruction* instruction) const;

  std::optional<int64_t> ParameterNumber(
      const HloInstruction* instruction) const;

  bool DerivedInput(const HloInstruction* instruction) const;

  const Entry& Get(const HloInstruction* instruction) const;

  void Add(const InstructionProperties& other);

  void AddDefaultProperties(const HloInstruction* instruction);

  bool AllowRecomputation(const HloInstruction* instruction) const;

  void AddBackwardAlias(HloInstruction* instruction, HloInstruction* alias);

  void AddForwardAlias(HloInstruction* instruction, HloInstruction* alias);

  HloInstruction* BackwardAlias(const HloInstruction* instruction) const;

  HloInstruction* ForwardAlias(const HloInstruction* instruction) const;

  void Clone(const HloInstruction* source, const HloInstruction* target);

  static InstructionProperties Create(HloModule* module);

  static InstructionProperties Create(HloComputation* computation);

 private:
  void AddBackwardAlias(HloInstruction* instruction, Entry& instruction_props,
                        HloInstruction* alias, Entry& alias_props);
  void AddForwardAlias(HloInstruction* instruction, Entry& instruction_props,
                       HloInstruction* alias, Entry& alias_props);

  void AddForwardProperties(HloComputation* computation);

  void AddBackwardProperties(HloComputation* computation, bool loop);

  void AddEntryProperties(HloComputation* computation);

  absl::flat_hash_map<const HloInstruction*, Entry> entries_;
};

absl::Status RemoveUnusedInstructions(HloComputation* computation);

absl::Status RemoveInstructionBackToParameters(HloComputation* computation,
                                               HloInstruction* instruction);

}  // namespace xla

#endif  // XLA_PJRT_LEGATE_MPMD_UTILS_H_
