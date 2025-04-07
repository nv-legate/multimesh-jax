/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/legate/mpmd_utils.h"

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

using ::testing::AllOf;
using ::testing::Field;
using ::testing::FieldsAre;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Ne;
using ::testing::Not;
using ::testing::Property;

class MpmdUtilsTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

class InstructionPropertiesMatcher
    : public ::testing::MatcherInterface<const std::pair<
          const HloInstruction* const, InstructionProperties::Entry>&> {
 public:
  using fxn = std::function<bool(const std::pair<const HloInstruction* const,
                                                 InstructionProperties::Entry>&,
                                 ::testing::MatchResultListener* listener)>;

  explicit InstructionPropertiesMatcher(std::string description, fxn f)
      : description_(std::move(description)),
        match_and_explain_(std::move(f)) {}

  bool MatchAndExplain(
      const std::pair<const HloInstruction* const,
                      InstructionProperties::Entry>& properties,
      ::testing::MatchResultListener* listener) const override {
    return match_and_explain_(properties, listener);
  }

  void DescribeTo(std::ostream* os) const override { *os << description_; }

 private:
  fxn match_and_explain_;
  std::string description_;
};

template <class Matcher>
auto OpMatches(Matcher&& matcher) {
  return Field(&std::pair<const HloInstruction* const,
                          InstructionProperties::Entry>::first,
               std::forward<Matcher>(matcher));
}

inline ::testing::Matcher<
    const std::pair<const HloInstruction* const, InstructionProperties::Entry>&>
HasBackwardAlias() {
  return ::testing::MakeMatcher(new InstructionPropertiesMatcher(
      absl::StrCat("instruction has backward alias"),
      [](const std::pair<const HloInstruction* const,
                         InstructionProperties::Entry>& properties,
         ::testing::MatchResultListener* listener) {
        return properties.second.backward_alias != nullptr;
      }));
}

inline ::testing::Matcher<
    const std::pair<const HloInstruction* const, InstructionProperties::Entry>&>
HasForwardAlias() {
  return ::testing::MakeMatcher(new InstructionPropertiesMatcher(
      absl::StrCat("instruction has forward alias"),
      [](const std::pair<const HloInstruction* const,
                         InstructionProperties::Entry>& properties,
         ::testing::MatchResultListener* listener) {
        return properties.second.forward_alias != nullptr;
      }));
}

inline ::testing::Matcher<
    const std::pair<const HloInstruction* const, InstructionProperties::Entry>&>
AdditiveToInput(std::optional<std::string> name = std::nullopt) {
  return ::testing::MakeMatcher(new InstructionPropertiesMatcher(
      absl::StrCat("instruction is additive to input"),
      [=](const std::pair<const HloInstruction* const,
                          InstructionProperties::Entry>& properties,
          ::testing::MatchResultListener* listener) {
        if (properties.second.additive_to_while_input.empty()) {
          *listener << properties.first->name()
                    << " is not additive to an input";
          return false;
        } else if (properties.second.additive_to_while_input.size() > 1) {
          *listener << properties.first->name()
                    << " is additive to multiple inputs";
          return false;
        } else if (name.has_value() &&
                   properties.second.additive_to_while_input.front()->name() !=
                       *name) {
          *listener
              << properties.first->name()
              << " has wrong additive input: " << *name << " != "
              << properties.second.additive_to_while_input.front()->name();
          return false;
        }
        return true;
      }));
}

inline ::testing::Matcher<
    const std::pair<const HloInstruction* const, InstructionProperties::Entry>&>
ElementwiseToOutput(std::optional<std::string> name = std::nullopt) {
  return ::testing::MakeMatcher(new InstructionPropertiesMatcher(
      absl::StrCat("instruction has forward alias"),
      [=](const std::pair<const HloInstruction* const,
                          InstructionProperties::Entry>& properties,
          ::testing::MatchResultListener* listener) {
        if (properties.second.elementwise_connected_to_entry_output ==
            nullptr) {
          *listener << properties.first->name()
                    << " is elementwise to an output";
          return false;
        } else if (name.has_value() &&
                   properties.second.elementwise_connected_to_entry_output
                           ->name() != *name) {
          *listener << properties.first->name()
                    << " has wrong additive input: " << *name << " != "
                    << properties.second.elementwise_connected_to_entry_output
                           ->name();
          return false;
        }
        return true;
      }));
}

inline ::testing::Matcher<
    const std::pair<const HloInstruction* const, InstructionProperties::Entry>&>
DerivedParameter() {
  return ::testing::MakeMatcher(new InstructionPropertiesMatcher(
      absl::StrCat("instruction is a derived constants"),
      [](const std::pair<const HloInstruction* const,
                         InstructionProperties::Entry>& properties,
         ::testing::MatchResultListener* listener) {
        return properties.second.derived_parameter;
      }));
}

inline ::testing::Matcher<
    const std::pair<const HloInstruction* const, InstructionProperties::Entry>&>
DerivedConstant() {
  return ::testing::MakeMatcher(new InstructionPropertiesMatcher(
      absl::StrCat("instruction is a derived constants"),
      [=](const std::pair<const HloInstruction* const,
                          InstructionProperties::Entry>& properties,
          ::testing::MatchResultListener* listener) {
        return properties.second.derived_constant;
      }));
}

inline ::testing::Matcher<
    const std::pair<const HloInstruction* const, InstructionProperties::Entry>&>
LoopCarriedIndex() {
  return ::testing::MakeMatcher(new InstructionPropertiesMatcher(
      absl::StrCat("instruction is a derived constants"),
      [](const std::pair<const HloInstruction* const,
                         InstructionProperties::Entry>& properties,
         ::testing::MatchResultListener* listener) {
        return properties.second.loop_carried_index.has_value();
      }));
}

template <class Matcher>
auto PropertyInstruction(Matcher&& matcher) {
  return Field(&std::pair<const HloInstruction* const,
                          InstructionProperties::Entry>::first,
               std::forward<Matcher>(matcher));
}

static constexpr absl::string_view kMultipleMicrobatchSlicesHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4]{0}, f32[4]{0})->f32[]}

%f.impl.11 (Arg_0.12: f32[2,4], Arg_1.13: f32[4]) -> f32[2,4] {
  %Arg_0.12 = f32[2,4]{1,0} parameter(0)
  %Arg_1.13 = f32[4]{0} parameter(1)
  %reshape.14 = f32[1,4]{1,0} reshape(f32[4]{0} %Arg_1.13)
  %broadcast.15 = f32[1,4]{1,0} broadcast(f32[1,4]{1,0} %reshape.14), dimensions={0,1}
  %reshape.16 = f32[4]{0} reshape(f32[1,4]{1,0} %broadcast.15)
  %broadcast.17 = f32[2,4]{1,0} broadcast(f32[4]{0} %reshape.16), dimensions={1}
  ROOT %multiply.18 = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %Arg_0.12, f32[2,4]{1,0} %broadcast.17)
}

%f.impl.19 (Arg_0.20: f32[2,4], Arg_1.21: f32[4]) -> f32[2,4] {
  %Arg_0.20 = f32[2,4]{1,0} parameter(0)
  %Arg_1.21 = f32[4]{0} parameter(1)
  %reshape.22 = f32[1,4]{1,0} reshape(f32[4]{0} %Arg_1.21)
  %broadcast.23 = f32[1,4]{1,0} broadcast(f32[1,4]{1,0} %reshape.22), dimensions={0,1}
  %reshape.24 = f32[4]{0} reshape(f32[1,4]{1,0} %broadcast.23)
  %broadcast.25 = f32[2,4]{1,0} broadcast(f32[4]{0} %reshape.24), dimensions={1}
  ROOT %multiply.26 = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %Arg_0.20, f32[2,4]{1,0} %broadcast.25)
}

%region_1.27 (Arg_0.28: f32[], Arg_1.29: f32[]) -> f32[] {
  %Arg_0.28 = f32[] parameter(0)
  %Arg_1.29 = f32[] parameter(1)
  ROOT %add.30 = f32[] add(f32[] %Arg_0.28, f32[] %Arg_1.29)
}

%g.impl.31 (Arg_0.32: f32[2,4], Arg_1.33: f32[4]) -> f32[] {
  %Arg_0.32 = f32[2,4]{1,0} parameter(0)
  %Arg_1.33 = f32[4]{0} parameter(1)
  %reshape.35 = f32[1,4]{1,0} reshape(f32[4]{0} %Arg_1.33)
  %broadcast.36 = f32[1,4]{1,0} broadcast(f32[1,4]{1,0} %reshape.35), dimensions={0,1}
  %reshape.37 = f32[4]{0} reshape(f32[1,4]{1,0} %broadcast.36)
  %broadcast.38 = f32[2,4]{1,0} broadcast(f32[4]{0} %reshape.37), dimensions={1}
  %multiply.39 = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %Arg_0.32, f32[2,4]{1,0} %broadcast.38)
  %constant.34 = f32[] constant(0)
  ROOT %reduce.40 = f32[] reduce(f32[2,4]{1,0} %multiply.39, f32[] %constant.34), dimensions={0,1}, to_apply=%region_1.27
}

%region_1.41 (Arg_0.42: f32[], Arg_1.43: f32[]) -> f32[] {
  %Arg_0.42 = f32[] parameter(0)
  %Arg_1.43 = f32[] parameter(1)
  ROOT %add.44 = f32[] add(f32[] %Arg_0.42, f32[] %Arg_1.43)
}

%g.impl.45 (Arg_0.46: f32[2,4], Arg_1.47: f32[4]) -> f32[] {
  %Arg_0.46 = f32[2,4]{1,0} parameter(0)
  %Arg_1.47 = f32[4]{0} parameter(1)
  %reshape.49 = f32[1,4]{1,0} reshape(f32[4]{0} %Arg_1.47)
  %broadcast.50 = f32[1,4]{1,0} broadcast(f32[1,4]{1,0} %reshape.49), dimensions={0,1}
  %reshape.51 = f32[4]{0} reshape(f32[1,4]{1,0} %broadcast.50)
  %broadcast.52 = f32[2,4]{1,0} broadcast(f32[4]{0} %reshape.51), dimensions={1}
  %multiply.53 = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %Arg_0.46, f32[2,4]{1,0} %broadcast.52)
  %constant.48 = f32[] constant(0)
  ROOT %reduce.54 = f32[] reduce(f32[2,4]{1,0} %multiply.53, f32[] %constant.48), dimensions={0,1}, to_apply=%region_1.41
}

%None.55 (Arg_0.56: f32[4,4], Arg_1.57: f32[4,4], Arg_2.58: f32[4], Arg_3.59: f32[4], Arg_4.60: s32[], Arg_5.61: f32[]) -> (s32[], f32[]) {
  %Arg_0.56 = f32[4,4]{1,0} parameter(0)
  %Arg_4.60 = s32[] parameter(4)
  %constant.64 = s32[] constant(0)
  %compare.65 = pred[] compare(s32[] %Arg_4.60, s32[] %constant.64), direction=LT
  %constant.63 = s32[] constant(4)
  %add.66 = s32[] add(s32[] %Arg_4.60, s32[] %constant.63)
  %select.67 = s32[] select(pred[] %compare.65, s32[] %add.66, s32[] %Arg_4.60)
  %dynamic-slice.68 = f32[2,4]{1,0} dynamic-slice(f32[4,4]{1,0} %Arg_0.56, s32[] %select.67, s32[] %constant.64), dynamic_slice_sizes={2,4}
  %custom-call.69 = f32[2,4]{1,0} custom-call(f32[2,4]{1,0} %dynamic-slice.68), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  %Arg_1.57 = f32[4,4]{1,0} parameter(1)
  %compare.70 = pred[] compare(s32[] %Arg_4.60, s32[] %constant.64), direction=LT
  %add.71 = s32[] add(s32[] %Arg_4.60, s32[] %constant.63)
  %select.72 = s32[] select(pred[] %compare.70, s32[] %add.71, s32[] %Arg_4.60)
  %dynamic-slice.73 = f32[2,4]{1,0} dynamic-slice(f32[4,4]{1,0} %Arg_1.57, s32[] %select.72, s32[] %constant.64), dynamic_slice_sizes={2,4}
  %custom-call.74 = f32[2,4]{1,0} custom-call(f32[2,4]{1,0} %dynamic-slice.73), custom_call_target="MicrobatchSlice", backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  %multiply.76 = f32[2,4]{1,0} multiply(f32[2,4]{1,0} %custom-call.69, f32[2,4]{1,0} %custom-call.74)
  %Arg_2.58 = f32[4]{0} parameter(2)
  %call.77 = f32[2,4]{1,0} call(f32[2,4]{1,0} %multiply.76, f32[4]{0} %Arg_2.58), to_apply=%f.impl.11
  %custom-call.78 = f32[2,4]{1,0} call(f32[2,4]{1,0} %multiply.76, f32[4]{0} %Arg_2.58), to_apply=%f.impl.19
  %Arg_3.59 = f32[4]{0} parameter(3)
  %call.79 = f32[] call(f32[2,4]{1,0} %custom-call.78, f32[4]{0} %Arg_3.59), to_apply=%g.impl.31
  %constant.62 = s32[] constant(2)
  %add.75 = s32[] add(s32[] %Arg_4.60, s32[] %constant.62)
  %Arg_5.61 = f32[] parameter(5)
  %custom-call.80 = f32[] call(f32[2,4]{1,0} %custom-call.78, f32[4]{0} %Arg_3.59), to_apply=%g.impl.45
  %add.81 = f32[] add(f32[] %Arg_5.61, f32[] %custom-call.80)
  ROOT %tuple.82 = (s32[], f32[]) tuple(s32[] %add.75, f32[] %add.81)
}

%region_0.83 (arg_tuple.84: (s32[], s32[], f32[], f32[4,4], f32[4,4], /*index=5*/f32[4], f32[4])) -> (s32[], s32[], f32[], f32[4,4], f32[4,4], /*index=5*/f32[4], f32[4]) {
  %arg_tuple.84 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) parameter(0)
  %get-tuple-element.85 = s32[] get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.84), index=0
  %constant.92 = s32[] constant(1)
  %add.96 = s32[] add(s32[] %get-tuple-element.85, s32[] %constant.92)
  %get-tuple-element.88 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.84), index=3
  %get-tuple-element.89 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.84), index=4
  %get-tuple-element.90 = f32[4]{0} get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.84), index=5
  %get-tuple-element.91 = f32[4]{0} get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.84), index=6
  %get-tuple-element.86 = s32[] get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.84), index=1
  %get-tuple-element.87 = f32[] get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.84), index=2
  %call.93 = (s32[], f32[]) call(f32[4,4]{1,0} %get-tuple-element.88, f32[4,4]{1,0} %get-tuple-element.89, f32[4]{0} %get-tuple-element.90, f32[4]{0} %get-tuple-element.91, s32[] %get-tuple-element.86, /*index=5*/f32[] %get-tuple-element.87), to_apply=%None.55
  %get-tuple-element.94 = s32[] get-tuple-element((s32[], f32[]) %call.93), index=0
  %get-tuple-element.95 = f32[] get-tuple-element((s32[], f32[]) %call.93), index=1
  ROOT %tuple.97 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) tuple(s32[] %add.96, s32[] %get-tuple-element.94, f32[] %get-tuple-element.95, f32[4,4]{1,0} %get-tuple-element.88, f32[4,4]{1,0} %get-tuple-element.89, /*index=5*/f32[4]{0} %get-tuple-element.90, f32[4]{0} %get-tuple-element.91)
}

%region_2.98 (arg_tuple.99: (s32[], s32[], f32[], f32[4,4], f32[4,4], /*index=5*/f32[4], f32[4])) -> pred[] {
  %arg_tuple.99 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) parameter(0)
  %get-tuple-element.101 = s32[] get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.99), index=1
  %get-tuple-element.102 = f32[] get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.99), index=2
  %get-tuple-element.103 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.99), index=3
  %get-tuple-element.104 = f32[4,4]{1,0} get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.99), index=4
  %get-tuple-element.105 = f32[4]{0} get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.99), index=5
  %get-tuple-element.106 = f32[4]{0} get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.99), index=6
  %get-tuple-element.100 = s32[] get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %arg_tuple.99), index=0
  %constant.107 = s32[] constant(2)
  ROOT %compare.108 = pred[] compare(s32[] %get-tuple-element.100, s32[] %constant.107), direction=LT
}

ENTRY %main.117 (Arg_0.1: f32[4,4], Arg_1.2: f32[4,4], Arg_2.3: f32[4], Arg_3.4: f32[4]) -> f32[] {
  %constant.5 = s32[] constant(0)
  %copy.1 = s32[] copy(s32[] %constant.5)
  %copy.2 = s32[] copy(s32[] %constant.5)
  %constant.6 = f32[] constant(0)
  %copy = f32[] copy(f32[] %constant.6), backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  %Arg_0.1 = f32[4,4]{1,0} parameter(0)
  %Arg_1.2 = f32[4,4]{1,0} parameter(1)
  %Arg_2.3 = f32[4]{0} parameter(2)
  %Arg_3.4 = f32[4]{0} parameter(3)
  %tuple.10 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) tuple(s32[] %copy.1, s32[] %copy.2, f32[] %copy, f32[4,4]{1,0} %Arg_0.1, f32[4,4]{1,0} %Arg_1.2, /*index=5*/f32[4]{0} %Arg_2.3, f32[4]{0} %Arg_3.4)
  %while.109 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) while((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %tuple.10), condition=%region_2.98, body=%region_0.83, backend_config={"num_microbatches": 2, "slice_dim": 0, "size": 2, "batch_dim": 0}
  ROOT %get-tuple-element.112 = f32[] get-tuple-element((s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) %while.109), index=2
}
)";

TEST_F(MpmdUtilsTest, MicrobatchLoop) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kMultipleMicrobatchSlicesHlo, /*num_devies=*/1));

  auto properties = InstructionProperties::Create(module.get());

  // 4 parameters in entry, 4 parameter in while, 4 parameters in call, 2 in
  // each subcall
  EXPECT_THAT(properties.map(), Contains(DerivedParameter()).Times(Ge(16)));

  // 3 loop-carried accumulators that appear in 3 different spots + 1 root
  EXPECT_THAT(properties.map(), Contains(LoopCarriedIndex()).Times(12));

  EXPECT_THAT(properties.map(),
              AllOf(Contains(HasBackwardAlias()).Times(Gt(2)),
                    Contains(HasForwardAlias()).Times(Gt(2))));
}

static constexpr absl::string_view kNestedWhileHlo = R"(
HloModule jit_c, entry_computation_layout={(f32[4,4]{1,0}, f32[4,4]{1,0}, f32[4]{0}, f32[4]{0})->f32[]}

f.impl.11 {
  Arg_0.12 = f32[2,4]{1,0} parameter(0)
  Arg_1.13 = f32[4]{0} parameter(1)
  reshape.14 = f32[1,4]{1,0} reshape(Arg_1.13)
  broadcast.15 = f32[1,4]{1,0} broadcast(reshape.14), dimensions={0,1}
  reshape.16 = f32[4]{0} reshape(broadcast.15)
  broadcast.17 = f32[2,4]{1,0} broadcast(reshape.16), dimensions={1}
  ROOT multiply.18 = f32[2,4]{1,0} multiply(Arg_0.12, broadcast.17)
} // f.impl.11

f.impl.19 {
  Arg_0.20 = f32[2,4]{1,0} parameter(0)
  Arg_1.21 = f32[4]{0} parameter(1)
  reshape.22 = f32[1,4]{1,0} reshape(Arg_1.21)
  broadcast.23 = f32[1,4]{1,0} broadcast(reshape.22), dimensions={0,1}
  reshape.24 = f32[4]{0} reshape(broadcast.23)
  broadcast.25 = f32[2,4]{1,0} broadcast(reshape.24), dimensions={1}
  ROOT multiply.26 = f32[2,4]{1,0} multiply(Arg_0.20, broadcast.25)
} // f.impl.19

region_1.27 {
  Arg_0.28 = f32[] parameter(0)
  Arg_1.29 = f32[] parameter(1)
  ROOT add.30 = f32[] add(Arg_0.28, Arg_1.29)
}

g.impl.31 {
  Arg_0.32 = f32[2,4]{1,0} parameter(0)
  Arg_1.33 = f32[4]{0} parameter(1)
  reshape.35 = f32[1,4]{1,0} reshape(Arg_1.33)
  broadcast.36 = f32[1,4]{1,0} broadcast(reshape.35), dimensions={0,1}
  reshape.37 = f32[4]{0} reshape(broadcast.36)
  broadcast.38 = f32[2,4]{1,0} broadcast(reshape.37), dimensions={1}
  multiply.39 = f32[2,4]{1,0} multiply(Arg_0.32, broadcast.38)
  constant.34 = f32[] constant(0)
  ROOT reduce.40 = f32[] reduce(multiply.39, constant.34), dimensions={0,1}, to_apply=region_1.27
} // g.impl.31

region_1.41 {
  Arg_0.42 = f32[] parameter(0)
  Arg_1.43 = f32[] parameter(1)
  ROOT add.44 = f32[] add(Arg_0.42, Arg_1.43)
}

g.impl.45 {
  Arg_0.46 = f32[2,4]{1,0} parameter(0)
  Arg_1.47 = f32[4]{0} parameter(1)
  reshape.49 = f32[1,4]{1,0} reshape(Arg_1.47)
  broadcast.50 = f32[1,4]{1,0} broadcast(reshape.49), dimensions={0,1}
  reshape.51 = f32[4]{0} reshape(broadcast.50)
  broadcast.52 = f32[2,4]{1,0} broadcast(reshape.51), dimensions={1}
  multiply.53 = f32[2,4]{1,0} multiply(Arg_0.46, broadcast.52)
  constant.48 = f32[] constant(0)
  ROOT reduce.54 = f32[] reduce(multiply.53, constant.48), dimensions={0,1}, to_apply=region_1.41
} // g.impl.45

None.55 {
  Arg_0.56 = f32[4,4]{1,0} parameter(0)
  Arg_4.60 = s32[] parameter(4)
  constant.64 = s32[] constant(0)
  compare.65 = pred[] compare(Arg_4.60, constant.64), direction=LT
  constant.63 = s32[] constant(4)
  add.66 = s32[] add(Arg_4.60, constant.63)
  select.67 = s32[] select(compare.65, add.66, Arg_4.60)
  dynamic-slice.68 = f32[2,4]{1,0} dynamic-slice(Arg_0.56, select.67, constant.64), dynamic_slice_sizes={2,4}
  Arg_1.57 = f32[4,4]{1,0} parameter(1)
  compare.70 = pred[] compare(Arg_4.60, constant.64), direction=LT
  add.71 = s32[] add(Arg_4.60, constant.63)
  select.72 = s32[] select(compare.70, add.71, Arg_4.60)
  dynamic-slice.73 = f32[2,4]{1,0} dynamic-slice(Arg_1.57, select.72, constant.64), dynamic_slice_sizes={2,4}
  multiply.76 = f32[2,4]{1,0} multiply(dynamic-slice.68, dynamic-slice.73)
  Arg_2.58 = f32[4]{0} parameter(2)
  call.77 = f32[2,4]{1,0} call(multiply.76, Arg_2.58), to_apply=f.impl.11
  call.78 = f32[2,4]{1,0} call(multiply.76, Arg_2.58), to_apply=f.impl.19
  Arg_3.59 = f32[4]{0} parameter(3)
  call.79 = f32[] call(call.78, Arg_3.59), to_apply=g.impl.31
  constant.62 = s32[] constant(2)
  add.75 = s32[] add(Arg_4.60, constant.62)
  Arg_5.61 = f32[] parameter(5)
  call.80 = f32[] call(call.78, Arg_3.59), to_apply=g.impl.45
  add.81 = f32[] add(Arg_5.61, call.80)
  ROOT tuple.82 = (s32[], f32[]) tuple(add.75, add.81)
} // None.55

region_0.83 {
  arg_tuple.84 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) parameter(0)
  get-tuple-element.85 = s32[] get-tuple-element(arg_tuple.84), index=0
  constant.92 = s32[] constant(1)
  add.96 = s32[] add(get-tuple-element.85, constant.92)
  get-tuple-element.88 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=3
  get-tuple-element.89 = f32[4,4]{1,0} get-tuple-element(arg_tuple.84), index=4
  get-tuple-element.90 = f32[4]{0} get-tuple-element(arg_tuple.84), index=5
  get-tuple-element.91 = f32[4]{0} get-tuple-element(arg_tuple.84), index=6
  get-tuple-element.86 = s32[] get-tuple-element(arg_tuple.84), index=1
  get-tuple-element.87 = f32[] get-tuple-element(arg_tuple.84), index=2
  call.93 = (s32[], f32[]) call(get-tuple-element.88, get-tuple-element.89, get-tuple-element.90, get-tuple-element.91, get-tuple-element.86, /*index=5*/get-tuple-element.87), to_apply=None.55
  get-tuple-element.94 = s32[] get-tuple-element(call.93), index=0
  get-tuple-element.95 = f32[] get-tuple-element(call.93), index=1
  ROOT tuple.97 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) tuple(add.96, get-tuple-element.94, get-tuple-element.95, get-tuple-element.88, get-tuple-element.89, /*index=5*/get-tuple-element.90, get-tuple-element.91)
} // region_0.83

region_2.98 {
  arg_tuple.99 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) parameter(0)
  get-tuple-element.101 = s32[] get-tuple-element(arg_tuple.99), index=1
  get-tuple-element.102 = f32[] get-tuple-element(arg_tuple.99), index=2
  get-tuple-element.103 = f32[4,4]{1,0} get-tuple-element(arg_tuple.99), index=3
  get-tuple-element.104 = f32[4,4]{1,0} get-tuple-element(arg_tuple.99), index=4
  get-tuple-element.105 = f32[4]{0} get-tuple-element(arg_tuple.99), index=5
  get-tuple-element.106 = f32[4]{0} get-tuple-element(arg_tuple.99), index=6
  get-tuple-element.100 = s32[] get-tuple-element(arg_tuple.99), index=0
  constant.107 = s32[] constant(2)
  ROOT compare.108 = pred[] compare(get-tuple-element.100, constant.107), direction=LT
} // region_2.98

ENTRY main.117 {
  constant.5 = s32[] constant(0)
  copy.5 = f32[] copy(constant.5)
  copy.55 = f32[] copy(constant.5)
  constant.6 = f32[] constant(0)
  convert.6 = bf16[] convert(constant.6)
  broadcast.6 = bf16[4] broadcast(convert.6), dimensions={}
  copy.7 = f32[] copy(constant.6)
  Arg_0.1 = f32[4,4]{1,0} parameter(0)
  Arg_1.2 = f32[4,4]{1,0} parameter(1)
  Arg_2.3 = f32[4]{0} parameter(2)
  Arg_3.4 = f32[4]{0} parameter(3)
  tuple.10 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) tuple(copy.5, copy.55, copy.7, Arg_0.1, Arg_1.2, /*index=5*/Arg_2.3, Arg_3.4)
  while.109 = (s32[], s32[], f32[], f32[4,4]{1,0}, f32[4,4]{1,0}, /*index=5*/f32[4]{0}, f32[4]{0}) while(tuple.10), condition=region_2.98, body=region_0.83
  ROOT get-tuple-element.112 = f32[] get-tuple-element(while.109), index=2
} // main.117
)";

TEST_F(MpmdUtilsTest, BasicLoop) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kNestedWhileHlo, /*num_devices=*/1));

  auto properties = InstructionProperties::Create(module.get());
  // There should be two derived constants that are not constants directly
  EXPECT_THAT(properties.map(),
              Contains(AllOf(PropertyInstruction(Not(op::Constant())),
                             DerivedConstant()))
                  .Times(2));

  auto* param = module->entry_computation()
                    ->GetInstructionWithName("while.109")
                    ->called_computations()[0]
                    ->GetInstructionWithName("call.93")
                    ->called_computations()[0]
                    ->GetInstructionWithName("call.79")
                    ->called_computations()[0]
                    ->GetInstructionWithName("Arg_1.33");
  ASSERT_NE(param, nullptr);

  int count = 0;
  properties.ForEachBackwardAlias(param, [&](HloInstruction* i) { ++count; });
  EXPECT_EQ(count, 3);

  auto* output = module->entry_computation()
                     ->GetInstructionWithName("while.109")
                     ->called_computations()[0]
                     ->GetInstructionWithName("call.93")
                     ->called_computations()[0]
                     ->GetInstructionWithName("add.81");
  count = 0;

  ASSERT_NE(output, nullptr);
  properties.ForEachForwardAlias(output, [&](HloInstruction* i) { ++count; });
  EXPECT_EQ(count, 2);
}

TEST_F(MpmdUtilsTest, PreArgumentCompute) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("argument_recompute.txt", /*num_devices=*/1));

  auto properties = InstructionProperties::Create(module.get());
}

constexpr absl::string_view kElementwiseToOutputHlo = R"(
region_1.19 {
  Arg_0.20 = s32[] parameter(0)
  Arg_1.21 = s32[] parameter(1)
  ROOT add.22 = s32[] add(Arg_0.20, Arg_1.21)
}

region_0.23 {
  arg_tuple.24 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) parameter(0)
  get-tuple-element.25 = s32[] get-tuple-element(arg_tuple.24), index=0
  constant.31 = s32[] constant(1)
  add.61 = s32[] add(get-tuple-element.25, constant.31)
  get-tuple-element.26 = s32[] get-tuple-element(arg_tuple.24), index=1
  constant.32 = s32[] constant(2)
  add.50 = s32[] add(get-tuple-element.26, constant.32)
  get-tuple-element.27 = s32[4]{0} get-tuple-element(arg_tuple.24), index=2
  get-tuple-element.28 = s32[4,4]{1,0} get-tuple-element(arg_tuple.24), index=3
  constant.34 = s32[] constant(0)
  compare.35 = pred[] compare(get-tuple-element.26, constant.34), direction=LT
  constant.33 = s32[] constant(4)
  add.36 = s32[] add(get-tuple-element.26, constant.33)
  select.37 = s32[] select(compare.35, add.36, get-tuple-element.26)
  dynamic-slice.38 = s32[2,4]{1,0} dynamic-slice(get-tuple-element.28, select.37, constant.34), dynamic_slice_sizes={2,4}
  get-tuple-element.29 = s32[4,1]{1,0} get-tuple-element(arg_tuple.24), index=4
  compare.40 = pred[] compare(get-tuple-element.26, constant.34), direction=LT
  add.41 = s32[] add(get-tuple-element.26, constant.33)
  select.42 = s32[] select(compare.40, add.41, get-tuple-element.26)
  dynamic-slice.43 = s32[2,1]{1,0} dynamic-slice(get-tuple-element.29, select.42, constant.34), dynamic_slice_sizes={2,1}
  broadcast.51 = s32[2,1]{1,0} broadcast(dynamic-slice.43), dimensions={0,1}
  reshape.52 = s32[2]{0} reshape(broadcast.51)
  broadcast.53 = s32[2,4]{1,0} broadcast(reshape.52), dimensions={0}
  add.54 = s32[2,4]{1,0} add(dynamic-slice.38, broadcast.53)
  get-tuple-element.30 = s32[4,1]{1,0} get-tuple-element(arg_tuple.24), index=5
  compare.45 = pred[] compare(get-tuple-element.26, constant.34), direction=LT
  add.46 = s32[] add(get-tuple-element.26, constant.33)
  select.47 = s32[] select(compare.45, add.46, get-tuple-element.26)
  dynamic-slice.48 = s32[2,1]{1,0} dynamic-slice(get-tuple-element.30, select.47, constant.34), dynamic_slice_sizes={2,1}
  broadcast.55 = s32[2,1]{1,0} broadcast(dynamic-slice.48), dimensions={0,1}
  reshape.56 = s32[2]{0} reshape(broadcast.55)
  broadcast.57 = s32[2,4]{1,0} broadcast(reshape.56), dimensions={0}
  add.58 = s32[2,4]{1,0} add(add.54, broadcast.57)
  reduce.59 = s32[4]{0} reduce(add.58, constant.34), dimensions={0}, to_apply=region_1.19
  add.60 = s32[4]{0} add(get-tuple-element.27, reduce.59)
  reshape.60 = s32[4]{0} reshape(add.60)
  convert.60 = s32[4]{0} convert(reshape.60)
  ROOT tuple.62 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) tuple(add.61, add.50, convert.60, get-tuple-element.28, get-tuple-element.29, /*index=5*/get-tuple-element.30)
} // region_0.23

region_2.63 {
  arg_tuple.64 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) parameter(0)
  get-tuple-element.66 = s32[] get-tuple-element(arg_tuple.64), index=1
  get-tuple-element.67 = s32[4]{0} get-tuple-element(arg_tuple.64), index=2
  get-tuple-element.68 = s32[4,4]{1,0} get-tuple-element(arg_tuple.64), index=3
  get-tuple-element.69 = s32[4,1]{1,0} get-tuple-element(arg_tuple.64), index=4
  get-tuple-element.70 = s32[4,1]{1,0} get-tuple-element(arg_tuple.64), index=5
  get-tuple-element.65 = s32[] get-tuple-element(arg_tuple.64), index=0
  constant.71 = s32[] constant(2)
  ROOT compare.72 = pred[] compare(get-tuple-element.65, constant.71), direction=LT
} // region_2.63

region_3.80 {
  Arg_0.81 = s32[] parameter(0)
  Arg_1.82 = s32[] parameter(1)
  ROOT add.83 = s32[] add(Arg_0.81, Arg_1.82)
}

ENTRY main.87 {
  constant.10 = s32[] constant(0)
  constant.4 = s32[] constant(0)
  broadcast.5 = s32[4]{0} broadcast(constant.4), dimensions={}
  Arg_0.1 = s32[4,4]{1,0} parameter(0), sharding={replicated}
  constant.8 = s32[] constant(2)
  broadcast.9 = s32[4,4]{1,0} broadcast(constant.8), dimensions={}
  multiply.11 = s32[4,4]{1,0} multiply(Arg_0.1, broadcast.9)
  Arg_1.2 = s32[4,1]{1,0} parameter(1), sharding={replicated}
  constant.6 = s32[] constant(2)
  broadcast.7 = s32[4,1]{1,0} broadcast(constant.6), dimensions={}
  multiply.12 = s32[4,1]{1,0} multiply(Arg_1.2, broadcast.7)
  Arg_2.3 = s32[4,1]{1,0} parameter(2), sharding={replicated}
  multiply.13 = s32[4,1]{1,0} multiply(Arg_2.3, broadcast.7)
  tuple.18 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) tuple(constant.10, constant.10, broadcast.5, multiply.11, multiply.12, multiply.13)
  while.73 = (s32[], s32[], s32[4]{0}, s32[4,4]{1,0}, s32[4,1]{1,0}, /*index=5*/s32[4,1]{1,0}) while(tuple.18), condition=region_2.63, body=region_0.23
  get-tuple-element.74 = s32[] get-tuple-element(while.73), index=0
  get-tuple-element.75 = s32[] get-tuple-element(while.73), index=1
  get-tuple-element.77 = s32[4,4]{1,0} get-tuple-element(while.73), index=3
  get-tuple-element.78 = s32[4,1]{1,0} get-tuple-element(while.73), index=4
  get-tuple-element.79 = s32[4,1]{1,0} get-tuple-element(while.73), index=5
  get-tuple-element.76 = s32[4]{0} get-tuple-element(while.73), index=2
  reduce.84 = s32[] reduce(get-tuple-element.76, constant.10), dimensions={0}, to_apply=region_3.80
  broadcast.84 = s32[4,4] broadcast(reduce.84), dimensions={}
  add.0 = add(Arg_0.1, broadcast.84)
  add.1 = add(Arg_1.2, broadcast.84)
  add.2 = add(Arg_2.3, broadcast.84)
  ROOT get-tuple-element.86 = (s32[4,4], s32[4,4], s32[4,4]) tuple(add.0, add.1, add.2)
} // main.87
)";

TEST_F(MpmdUtilsTest, ElementwiseToOutput) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kElementwiseToOutputHlo, /*num_devices=*/1));
  auto properties = InstructionProperties::Create(module.get());

  EXPECT_THAT(
      properties.map(),
      AllOf(
          Contains(AllOf(ElementwiseToOutput(), OpMatches(op::Add()))).Times(3),
          Contains(AllOf(ElementwiseToOutput(), OpMatches(op::Parameter())))
              .Times(3)));

  EXPECT_THAT(properties.map(),
              AllOf(Contains(AllOf(AdditiveToInput("get-tuple-element.27"),
                                   OpMatches(op::Add()))),
                    Contains(AllOf(AdditiveToInput("get-tuple-element.27"),
                                   OpMatches(op::Reduce())))));
}

TEST_F(MpmdUtilsTest, FailedDotShardMap) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("failed_dot_shard_map.txt", /*num_devices=*/8));
  auto properties = InstructionProperties::Create(module.get());
  // TODO: add expects
}

TEST_F(MpmdUtilsTest, FailedReduceShardMap) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromPath("failed_shard_map_reduce.txt", /*num_devices=*/8));
  auto properties = InstructionProperties::Create(module.get());
  // TODO: add expects
}

}  // namespace
}  // namespace xla
