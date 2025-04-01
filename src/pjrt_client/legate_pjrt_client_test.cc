#include "xla/pjrt/legate/legate_pjrt_client.h"

#include "gmock/gmock.h"
#include "xla/pjrt/legate/hlo_partition.h"
#include "xla/pjrt/legate/legate_test_base.h"
#include "xla/pjrt/legate/mpmd_test_base.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

struct Config {
  bool use_module_config_auto_output_sharding = false;
  bool use_auto_input_sharding = false;
};

class LegateClientTest : public LegateTestBase {
 public:
  LegateClientTest() : LegateTestBase() {}

  void SetUp() override {
    LegateTestBase::SetUp();
    EnableLegateRecomputation(false);
  }
};

static constexpr absl::string_view kArgumentReshardingHlo = R"(
ENTRY main.23 {
  Arg_0.1 = f32[4,2]{1,0} parameter(0), sharding={devices=[2,1]0,1}
  constant.3 = f32[] constant(2)
  broadcast.4 = f32[4,2]{1,0} broadcast(constant.3), dimensions={}
  custom-call.15 = f32[4,2]{1,0} custom-call(Arg_0.1), custom_call_target="sharding", sharding={devices=[4,1]0,1,2,3}
  multiply.8 = f32[4,2]{1,0} multiply(custom-call.15, broadcast.4), sharding={devices=[4,1]0,1,2,3}
  Arg_1.2 = f32[4,2]{1,0} parameter(1), sharding={devices=[2,1]2,3}
  cosine.19 = f32[4,2]{1,0} cosine(Arg_1.2)
  custom-call.16 = f32[4,2]{1,0} custom-call(multiply.8), custom_call_target="sharding", sharding={devices=[2,1]2,3}
  add.20 = f32[4,2]{1,0} add(cosine.19, custom-call.16)
  ROOT tuple.42 = (f32[4,2]{1,0}, f32[4,2]{1,0}) tuple(add.20, add.20), sharding={{devices=[2,1]0,1},{devices=[2,1]2,3}}
} // main.23
)";
static constexpr absl::string_view kArgumentReshardingArg0ShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 1
tile_assignment_devices: 0
tile_assignment_devices: 1
)";
static constexpr absl::string_view kArgumentReshardingArg1ShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 2
tile_assignment_dimensions: 1
tile_assignment_devices: 2
tile_assignment_devices: 3
)";

MATCHER_P(ProtoMatchesSharding, sharding, "") {
  auto test_sharding = HloSharding::FromProto(arg);
  if (!test_sharding.ok()) {
    std::cerr << "cannot convert OpSharding to HloSharding" << std::endl;
    return false;
  }

  return *test_sharding == sharding;
}

MATCHER_P(ProtoMatchesShardingShape, sharding, "") {
  auto test_sharding = HloSharding::FromProto(arg);
  if (!test_sharding.ok()) {
    std::cerr << "cannot convert OpSharding to HloSharding" << std::endl;
    return false;
  }

  return test_sharding->tile_assignment().dimensions() ==
         sharding.tile_assignment().dimensions();
}

TEST_F(LegateClientTest, ArgumentReshard) {
  int num_devices = 4;
  if (Skip(num_devices)) {
    GTEST_SKIP() << "Skipping with too few devices";
  }
  TF_ASSERT_OK_AND_ASSIGN(auto exe, Compile(kArgumentReshardingHlo,
                                            /*num_devices=*/num_devices, {}));

  auto param_shardings = exe->GetParameterShardings();
  ASSERT_TRUE(param_shardings.has_value());

  TF_ASSERT_OK_AND_ASSIGN(auto Arg0_sharding,
                          GetSharding(kArgumentReshardingArg0ShardingPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(auto Arg1_sharding,
                          GetSharding(kArgumentReshardingArg1ShardingPbtxt));
  EXPECT_THAT(*param_shardings,
              ElementsAre(ProtoMatchesSharding(Arg0_sharding),
                          ProtoMatchesSharding(Arg1_sharding)));

  auto output_shardings = exe->GetOutputShardings();
  ASSERT_TRUE(output_shardings.has_value());
  // the only thing required of the outputs is that they have the right sharding
  // shape Legate doesn't actually care what devices the roots land on
  EXPECT_THAT(*output_shardings,
              ElementsAre(ProtoMatchesShardingShape(Arg0_sharding),
                          ProtoMatchesShardingShape(Arg1_sharding)));
}

static constexpr absl::string_view kAutoShardingReshardedParameterHlo = R"(
ENTRY main.23 {
  Arg_0.1 = f32[4,2]{1,0} parameter(0), sharding={devices=[4,1]0,1,2,3}
  Arg_1.2 = f32[4,2]{1,0} parameter(1), sharding={devices=[4,1]0,1,2,3}
  multiply.1 = f32[4,2]{1,0} multiply(Arg_1.2, Arg_1.2), metadata={op_name="layer0"}
  custom-call.15 = f32[4,2]{1,0} custom-call(Arg_0.1), custom_call_target="AutoSharding", backend_config={"axes": [["x"], ["y"]]}
  custom-call.16 = f32[4,2]{1,0} custom-call(multiply.1), custom_call_target="AutoSharding", backend_config={"axes": [["x"], ["y"]]}
  multiply.8 = f32[4,2]{1,0} multiply(custom-call.15, custom-call.16), sharding={devices=[2,1]0,1}, metadata={op_name="layer1"}
  custom-call.17 = f32[4,2]{1,0} custom-call(multiply.8), custom_call_target="AutoSharding", backend_config={"axes": [["x"], ["y"]]}
  ROOT cosine.9 = f32[4,2]{1,0} cosine(multiply.8), sharding={devices=[4,1]0,1,2,3}, metadata={op_name="layer2"}
} // main.23
)";
static constexpr absl::string_view kAutoShardingParameterHloShardingPbtxt = R"(
type: OTHER
tile_assignment_dimensions: 4
tile_assignment_dimensions: 1
tile_assignment_devices: 0
tile_assignment_devices: 1
tile_assignment_devices: 2
tile_assignment_devices: 3
)";

TEST_F(LegateClientTest, AutoShardingReshardedParameter) {
  RegisterMatcherTestTask("layer0", {0, 1, 2, 3}, {4, 1}, {"x", "y"},
                          {{"x", "x"}, {"y", "y"}});
  RegisterMatcherTestTask("layer1", {0, 1}, {2, 1}, {"x", "y"},
                          {{"x", "x"}, {"y", "y"}});
  RegisterMatcherTestTask("layer2", {0, 1, 2, 3}, {4, 1}, {"x", "y"},
                          {{"x", "x"}, {"y", "y"}});

  int num_devices = 4;
  if (Skip(num_devices)) {
    GTEST_SKIP() << "Skipping with too few devices";
  }
  TF_ASSERT_OK_AND_ASSIGN(auto exe, Compile(kAutoShardingReshardedParameterHlo,
                                            /*num_devices=*/num_devices,
                                            {.use_auto_input_sharding = true}));

  auto param_shardings = exe->GetParameterShardings();
  ASSERT_TRUE(param_shardings.has_value());

  // the parameters will get resharded across the tasks, but the parameter
  // sharding must be reported as the original, explicit shardings
  TF_ASSERT_OK_AND_ASSIGN(HloSharding arg_sharding,
                          GetSharding(kAutoShardingParameterHloShardingPbtxt));

  EXPECT_THAT(*param_shardings,
              ElementsAre(ProtoMatchesSharding(arg_sharding),
                          ProtoMatchesSharding(arg_sharding)));
}

}  // namespace
}  // namespace xla