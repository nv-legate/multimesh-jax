/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef XLA_PJRT_MULTIMESH_MPMD_TEST_BASE_H_
#define XLA_PJRT_MULTIMESH_MPMD_TEST_BASE_H_

#include <filesystem>
#include <vector>

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/multimesh/hlo_partition.h"
#include "xla/pjrt/multimesh/mpmd_partition.h"
#include "xla/pjrt/multimesh/mpmd_store.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {

struct MpmdTestConfig {
  bool use_module_config_auto_output_sharding = false;
  bool use_module_config_auto_param_sharding = false;
  bool use_auto_input_sharding = false;
  std::optional<int64_t> replicated_parameter_num_elements_cutoff{std::nullopt};
  std::optional<int64_t> recompute_from_arguments_if_cost_less_than{
      std::nullopt};
  bool run_simplification_passes = false;
  bool only_fuse_loop_tasks = false;
};

class MpmdTestBase : public HloTestBase {
 public:
  MpmdTestBase();

  void SetUp() override;

  void TearDown() override;

  absl::Status VerifyHloModule(const HloModule& module,
                               const HloPartition& partition);

  absl::StatusOr<std::unique_ptr<HloModule>> GetHloModuleFromPath(
      absl::string_view hlo_text_path, int num_devices);

  absl::StatusOr<std::unique_ptr<HloModule>> GetHloModuleFromText(
      absl::string_view hlo_text, int num_devices);

  absl::StatusOr<std::pair<std::vector<SpmdHloModuleTask>, int64_t>>
  RunMpmdOnHloString(absl::string_view hlo_string, int num_devices,
                     MpmdTestConfig cfg = MpmdTestConfig{});

  absl::StatusOr<std::pair<std::vector<SpmdHloModuleTask>, int64_t>>
  RunMpmdOnHloModule(std::unique_ptr<HloModule> module, int num_devices,
                     MpmdTestConfig cfg = MpmdTestConfig{});

  absl::StatusOr<std::pair<std::vector<SpmdHloModuleTask>, int64_t>>
  RunMpmdOnHloTextPath(absl::string_view hlo_text_path, int num_devices,
                       MpmdTestConfig cfg = MpmdTestConfig{});

  absl::StatusOr<HloSharding> GetSharding(absl::string_view pbtxt);

  std::vector<HloInstruction*> EntryComputationCalls(HloModule* module);

  std::vector<HloComputation*> EntryComputationCalledComputations(
      HloModule* module);

  auto EntryComputationCallInstructionLists(HloModule* module) {
    std::vector<decltype(module->entry_computation()->instructions())> lists;
    for (auto* comp : EntryComputationCalledComputations(module)) {
      lists.push_back(comp->instructions());
    }
    return lists;
  }

 protected:
  std::unique_ptr<HloPartition> partition_;

 private:
  std::filesystem::path testdata_root_;
};

class MpmdHloInstructionMatcher
    : public ::testing::MatcherInterface<const HloInstruction*> {
 public:
  using fxn = std::function<bool(const HloInstruction*,
                                 ::testing::MatchResultListener* listener)>;

  explicit MpmdHloInstructionMatcher(std::string description, fxn f)
      : match_and_explain_(std::move(f)),
        description_(std::move(description)) {}

  bool MatchAndExplain(const HloInstruction* instruction,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(std::ostream* os) const override { *os << description_; }

 private:
  fxn match_and_explain_;
  std::string description_;
};

class MpmdTaskMatcher
    : public ::testing::MatcherInterface<const SpmdHloModuleTask&> {
 public:
  using fxn = std::function<bool(const SpmdHloModuleTask&,
                                 ::testing::MatchResultListener* listener)>;

  explicit MpmdTaskMatcher(std::string description, fxn f)
      : match_and_explain_(std::move(f)),
        description_(std::move(description)) {}

  bool MatchAndExplain(const SpmdHloModuleTask&,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(std::ostream* os) const override { *os << description_; }

 private:
  fxn match_and_explain_;
  std::string description_;
};

class MpmdStoreMatcher : public ::testing::MatcherInterface<const Store&> {
 public:
  using fxn = std::function<bool(const Store&,
                                 ::testing::MatchResultListener* listener)>;

  explicit MpmdStoreMatcher(std::string description, fxn f)
      : match_and_explain_(std::move(f)),
        description_(std::move(description)) {}

  bool MatchAndExplain(const Store&,
                       ::testing::MatchResultListener* listener) const override;

  void DescribeTo(std::ostream* os) const override { *os << description_; }

 private:
  fxn match_and_explain_;
  std::string description_;
};

namespace mpmd_matchers {

std::vector<std::vector<HloInstruction*>> CalledComputationInstructions(
    HloModule* module);

std::vector<HloInstruction*> EntryComputationCalls(HloModule* module);

std::vector<HloInstruction*> FlatInstructions(HloModule* module);

std::vector<HloInstruction*> FlatInstructionsWithoutReducesAndPredicates(
    HloModule* module);

inline auto MetaOp() {
  namespace op = ::xla::testing::opcode_matchers;
  return ::testing::AnyOf(op::Parameter(), op::GetTupleElement(), op::Tuple(),
                          op::OptimizationBarrier(), op::Constant());
}

inline auto TrivialOp() {
  namespace op = ::xla::testing::opcode_matchers;
  return ::testing::AnyOf(op::Parameter(), op::CustomCall(), op::Broadcast(),
                          op::Convert(), op::GetTupleElement(), op::Tuple(),
                          op::OptimizationBarrier(), op::Constant());
}

inline auto NontrivialOp() { return ::testing::Not(TrivialOp()); }

template <class Matcher>
auto TaskInstructions(const Matcher& matcher) {
  using ::testing::Field;
  using ::testing::Property;
  return Field(
      &SpmdHloModuleTask::module,
      Property(&std::shared_ptr<SpmdModule>::get,
               Field(&SpmdModule::module,
                     Property(&std::unique_ptr<HloModule>::get,
                              Property(&HloModule::entry_computation,
                                       Property(&HloComputation::instructions,
                                                matcher))))));
}

template <class Matcher>
auto TaskDevices(const Matcher& matcher) {
  using ::testing::Field;
  return Field(&SpmdHloModuleTask::device_assignment, matcher);
}

inline auto ShardingOrReplicated(const HloSharding& sharding) {
  namespace op = ::xla::testing::opcode_matchers;
  return AnyOf(op::Sharding(sharding), op::Sharding(HloSharding::Replicate()),
               op::NoSharding());
}

template <class Matcher>
auto TaskParameters(const Matcher& matcher) {
  using ::testing::Field;
  using ::testing::Property;
  return Field(
      &SpmdHloModuleTask::module,
      Property(
          &std::shared_ptr<SpmdModule>::get,
          Field(&SpmdModule::module,
                Property(
                    &std::unique_ptr<HloModule>::get,
                    Property(&HloModule::entry_computation,
                             Property(&HloComputation::parameter_instructions,
                                      matcher))))));
}

template <class Matcher>
auto TaskOutputs(const Matcher& matcher) {
  using ::testing::Field;
  using ::testing::Property;
  return Field(&SpmdHloModuleTask::outputs, matcher);
}

template <class Matcher>
auto TaskInputs(const Matcher& matcher) {
  using ::testing::Field;
  using ::testing::Property;
  return Field(&SpmdHloModuleTask::inputs, matcher);
}

template <class Matcher>
auto TaskRoots(const Matcher& matcher) {
  using ::testing::Field;
  using ::testing::Property;
  return Field(
      &SpmdHloModuleTask::module,
      Property(
          &std::shared_ptr<SpmdModule>::get,
          Field(&SpmdModule::module,
                Property(&std::unique_ptr<HloModule>::get,
                         Property(&HloModule::entry_computation,
                                  Property(&HloComputation::root_instruction,
                                           Property(&HloInstruction::operands,
                                                    matcher)))))));
}

std::vector<HloComputation*> WhileBodies(HloModule* module);

template <class Matcher>
auto MetadataSchedulingNames(const Matcher& matcher) {
  using ::testing::Field;
  using ::testing::Property;
  return Property(&HloInstruction::metadata,
                  Property(&OpMetadata::scheduling_name, matcher));
}

::testing::Matcher<const ::xla::HloInstruction*> TrivialShape();

::testing::Matcher<const ::xla::HloInstruction*> ShapeLargerThan(
    int64_t elements);

::testing::Matcher<const ::xla::HloInstruction*> HasColor();

::testing::Matcher<const ::xla::HloInstruction*> Color(std::string color);

::testing::Matcher<const ::xla::HloInstruction*> Devices(
    const HloPartition& partition, zuku::DeviceList devices);

template <class Matcher>
auto Users(const Matcher& matcher) {
  return Property(&HloInstruction::users, matcher);
}

template <class Matcher>
auto Operands(const Matcher& matcher) {
  return Property(&HloInstruction::operands, matcher);
}

::testing::Matcher<const ::xla::HloInstruction*> HasLogicalAxes();

::testing::Matcher<const SpmdHloModuleTask&> AnyTask();

::testing::Matcher<const SpmdHloModuleTask&> LoopTask(
    std::optional<bool> has_slice = std::nullopt);

struct MpmdStoreMatcherConfig {
  std::optional<Store::Type> type;
  std::optional<int> index;
  std::optional<HloSharding> sharding;
};

::testing::Matcher<const ::xla::Store&> Store(MpmdStoreMatcherConfig config);

}  // namespace mpmd_matchers

void RegisterNamedTestTask(
    std::string name, std::pair<int64_t, int64_t> devices,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes);

void RegisterMatcherTestTask(
    std::string matcher, std::pair<int64_t, int64_t> devices,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes);

void RegisterMatcherTestTaskWithFactory(
    std::string matcher,
    std::function<std::pair<std::pair<int64_t, int64_t>, std::string>(
        const std::string& task, bool backprop)>
        device_factory,
    std::vector<int64_t> dims, std::vector<std::string> axes,
    std::vector<std::pair<std::string, std::string>> logical_axes);

}  // namespace xla

#endif
