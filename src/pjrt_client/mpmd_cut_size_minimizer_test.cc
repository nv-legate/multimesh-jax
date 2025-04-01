#include "xla/pjrt/legate/mpmd_cut_size_minimizer.h"

#include "gmock/gmock.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

class MpmdCutSizeMinimizerTest : public MpmdTestBase {};

namespace op = ::xla::testing::opcode_matchers;
namespace m = ::xla::mpmd_matchers;

static constexpr absl::string_view kLargeDotWithSmallOperands = R"(
ENTRY main.32 {
  Arg_0.1 = f32[1024,8]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="red"}
  cos.0 = f32[1024,8]{1,0} cosine(Arg_0.1), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="red"}
  add.0 = f32[1024,8]{1,0} add(cos.0, Arg_0.1), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="red"}
  dot.0 = f32[1024,1024]{1,0} dot(add.0, cos.0), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[4,1]<=[4]}, frontend_attributes={color="red"}
  Arg_1.2 = f32[1024,1024]{1,0} parameter(1), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="blue"}
  cos.1 = f32[1024,1024]{1,0} cosine(dot.0), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="blue"}
  add.1 = f32[1024,1024]{1,0} add(cos.0, Arg_1.2), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="blue"}
  ROOT tuple = (f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) tuple(dot.0, add.1)
} // main.32
)";

TEST_F(MpmdCutSizeMinimizerTest, LargeDotWithSmallOperands) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module,
      GetHloModuleFromText(kLargeDotWithSmallOperands, /*num_devices=*/4));

  // before starting, the dots should be red and blue
  EXPECT_THAT(module->entry_computation()->instructions(),
              AllOf(Contains(AllOf(op::Dot(), m::Color("red"))).Times(1)));

  MpmdCutSizeMinimizer minimizer{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, minimizer.Run(module.get()));

  // all dots should be blue now
  EXPECT_THAT(module->entry_computation()->instructions(),
              AllOf(Not(Contains(AllOf(op::Dot(), m::Color("red")))),
                    Contains(AllOf(op::Dot(), m::Color("blue"))).Times(1)));
}

static constexpr absl::string_view kLargeDotTransposeWithSmallOperands = R"(
ENTRY main.32 {
  Arg_0.1 = f32[1024,8]{1,0} parameter(0), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="red"}
  cos.0 = f32[1024,8]{1,0} cosine(Arg_0.1), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="red"}
  add.0 = f32[1024,8]{1,0} add(cos.0, Arg_0.1), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="red"}
  dot.0 = f32[1024,1024]{1,0} dot(add.0, cos.0), lhs_contracting_dims={1}, rhs_contracting_dims={0}, sharding={devices=[4,1]<=[4]}, frontend_attributes={color="red"}
  transpose.0 = f32[1024,1024]{1,0} transpose(dot.0), dimensions={1,0}, frontend_attributes={color="red"}
  negate.0 = f32[1024,1024]{1,0} negate(transpose.0), frontend_attributes={color="red"}
  Arg_1.2 = f32[1024,1024]{1,0} parameter(1), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="blue"}
  cos.1 = f32[1024,1024]{1,0} cosine(negate.0), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="blue"}
  add.1 = f32[1024,1024]{1,0} add(cos.1, Arg_1.2), sharding={devices=[4,1]<=[4]}, frontend_attributes={color="blue"}
  ROOT tuple = (f32[1024,1024]{1,0}, f32[1024,1024]{1,0}) tuple(dot.0, add.1)
} // main.32
)";

// follow certain unary operands backward to see if their operands
// are candidates for minimizing cut size
TEST_F(MpmdCutSizeMinimizerTest, UnaryOperandMove) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto module, GetHloModuleFromText(kLargeDotTransposeWithSmallOperands,
                                        /*num_devices=*/4));

  // before starting, the dots should be red and blue
  EXPECT_THAT(module->entry_computation()->instructions(),
              AllOf(Contains(AllOf(op::Dot(), m::Color("red"))).Times(1)));

  MpmdCutSizeMinimizer minimizer{partition_.get()};
  TF_ASSERT_OK_AND_ASSIGN(bool changed, minimizer.Run(module.get()));

  // all dots should be blue now
  EXPECT_THAT(module->entry_computation()->instructions(),
              AllOf(Not(Contains(AllOf(op::Dot(), m::Color("red")))),
                    Contains(AllOf(op::Dot(), m::Color("blue"))).Times(1)));
}

}  // namespace
}  // namespace xla
