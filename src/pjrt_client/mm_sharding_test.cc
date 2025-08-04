/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "xla/pjrt/multimesh/mm_sharding.h"

#include "gmock/gmock.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "tsl/platform/protobuf.h"
#include "xla/util.h"

namespace xla {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Le;
using ::testing::Pointwise;

void PrintShardingIndices(const HloSharding& sharding) {
  std::cerr << sharding << std::endl;
  sharding.tile_assignment().array().Each([](auto indices, auto value) {
    std::cerr << value << " = (";
    for (auto idx : indices) {
      std::cerr << " " << idx;
    }
    std::cerr << " )" << std::endl;
  });
}

template <size_t N>
absl::InlinedVector<int64_t, N> ComputeIndexSet(
    int64_t global_index, const absl::InlinedVector<int64_t, N>& dims) {
  return xla::ComputeIndexSet<absl::InlinedVector<int64_t, N>>(global_index,
                                                               dims);
}

absl::StatusOr<Shape> GetShape(absl::string_view pbtxt) {
  ShapeProto shape_proto;
  if (!tsl::protobuf::TextFormat::ParseFromString(std::string(pbtxt),
                                                  &shape_proto)) {
    return InvalidArgument("failed to parse sharding proto");
  }
  return Shape(shape_proto);
}

absl::StatusOr<HloSharding> GetSharding(absl::string_view pbtxt) {
  OpSharding sharding_proto;
  if (!tsl::protobuf::TextFormat::ParseFromString(std::string(pbtxt),
                                                  &sharding_proto)) {
    return InvalidArgument("failed to parse sharding proto");
  }
  return HloSharding::FromProto(sharding_proto);
}

TEST(MultiMeshShardingTest, BasicIndexSetComputation) {
  {
    auto indices = ComputeIndexSet<4>(5, {2, 2, 2});
    EXPECT_THAT(indices, ElementsAre(1, 0, 1));
  }

  {
    auto indices = ComputeIndexSet<6>(13, {1, 2, 3, 4});
    EXPECT_THAT(indices, ElementsAre(0, 1, 0, 1));
  }

  {
    auto indices = ComputeIndexSet<6>(0, {});
    EXPECT_TRUE(indices.empty());
  }

  {
    auto indices = ComputeIndexSet<6>(0, {1, 1, 1, 1});
    EXPECT_THAT(indices, ElementsAre(0, 0, 0, 0));
  }
}

TEST(MultiMeshShardingTest, BasicGlobalIndexComputation) {
  {
    auto global_index = ComputeGlobalIndex<4, 4>({1, 0, 1}, {2, 2, 2});
    EXPECT_EQ(global_index, 5);
  }

  {
    auto global_index = ComputeGlobalIndex<4, 4>({0, 1, 0, 1}, {1, 2, 3, 4});
    EXPECT_EQ(global_index, 13);
  }

  {
    auto global_index = ComputeGlobalIndex<4, 4>({}, {});
    EXPECT_EQ(global_index, 0);
  }

  {
    auto global_index = ComputeGlobalIndex<4, 4>({0, 0, 0, 0}, {1, 1, 1, 1});
    EXPECT_EQ(global_index, 0);
  }
}

TEST(MultiMeshShardingTest, OneToOneLogicalToDeviceAxes) {
  zuku::DeviceList devices{{.start = 0, .num_devices = 8}};

  TF_ASSERT_OK_AND_ASSIGN(
      auto op_sharding,
      LogicalToPhysicalSharding({{0}, {1}, {2}}, devices, {2, 2, 2}));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          HloSharding::FromProto(op_sharding));

  EXPECT_THAT(hlo_sharding.tile_assignment().array(),
              ElementsAreArray(devices));
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 2, 2));
}

TEST(MultiMeshShardingTest, UnshardedLogicalAxis) {
  zuku::DeviceList devices{{.start = 0, .num_devices = 4}};

  TF_ASSERT_OK_AND_ASSIGN(
      auto op_sharding,
      LogicalToPhysicalSharding({{0}, {}, {1}}, devices, {2, 2}));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          HloSharding::FromProto(op_sharding));

  EXPECT_THAT(hlo_sharding.tile_assignment().array(),
              ElementsAreArray(devices));
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 1, 2));
}

TEST(MultiMeshShardingTest, ReorderedAxes) {
  zuku::DeviceList devices{{.start = 0, .num_devices = 6}};

  TF_ASSERT_OK_AND_ASSIGN(
      auto op_sharding,
      LogicalToPhysicalSharding({{1}, {}, {0}}, devices, {3, 2}));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          HloSharding::FromProto(op_sharding));

  EXPECT_THAT(hlo_sharding.tile_assignment().array(),
              ElementsAre(0, 2, 4, 1, 3, 5));
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(2, 1, 3));
}

TEST(MultiMeshShardingTest, DoubleLogicalAxisSharding) {
  zuku::DeviceList devices{{.start = 0, .num_devices = 12}};

  TF_ASSERT_OK_AND_ASSIGN(
      auto op_sharding,
      LogicalToPhysicalSharding({{0, 2}, {}, {1}}, devices, {3, 2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          HloSharding::FromProto(op_sharding));
  EXPECT_THAT(hlo_sharding.tile_assignment().array(),
              ElementsAre(0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11));
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(6, 1, 2));
}

TEST(MultiMeshShardingTest, SimpleReplicatedAxis) {
  zuku::DeviceList devices{{.start = 0, .num_devices = 6}};

  TF_ASSERT_OK_AND_ASSIGN(
      auto op_sharding,
      LogicalToPhysicalSharding({{}, {}, {0}}, devices, {3, 2}));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          HloSharding::FromProto(op_sharding));
  EXPECT_THAT(hlo_sharding.tile_assignment().array(),
              ElementsAre(0, 1, 2, 3, 4, 5));
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(),
              ElementsAre(1, 1, 3, 2));
  EXPECT_TRUE(op_sharding.replicate_on_last_tile_dim());
}

TEST(MultiMeshShardingTest, MoreDeviceAxesReplicated) {
  zuku::DeviceList devices{{.start = 0, .num_devices = 6}};

  TF_ASSERT_OK_AND_ASSIGN(auto op_sharding,
                          LogicalToPhysicalSharding({{0}}, devices, {3, 2}));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          HloSharding::FromProto(op_sharding));
  EXPECT_THAT(hlo_sharding.tile_assignment().array(),
              ElementsAre(0, 1, 2, 3, 4, 5));
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(), ElementsAre(3, 2));
  EXPECT_TRUE(op_sharding.replicate_on_last_tile_dim());
}

TEST(MultiMeshShardingTest, OutOfOrderReplicatedAxis) {
  zuku::DeviceList devices{{.start = 0, .num_devices = 6}};

  TF_ASSERT_OK_AND_ASSIGN(
      auto op_sharding,
      LogicalToPhysicalSharding({{}, {}, {1}}, devices, {3, 2}));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          HloSharding::FromProto(op_sharding));
  // { 0 0 0 0} -> { 0 0 }
  // { 0 0 0 1} -> { 1 0 }
  // { 0 0 0 2} -> { 2 0 }
  // { 0 0 1 0} -> { 0 1 }
  // { 0 0 1 1} -> { 1 1 }
  // { 0 0 1 2} -> { 2 1 }
  EXPECT_THAT(hlo_sharding.tile_assignment().array(),
              ElementsAre(0, 2, 4, 1, 3, 5));
  EXPECT_THAT(op_sharding.tile_assignment_dimensions(),
              ElementsAre(1, 1, 2, 3));
  EXPECT_TRUE(op_sharding.replicate_on_last_tile_dim());
}

TEST(MultiMeshShardingTest, FullyReplicated) {
  zuku::DeviceList devices{{.start = 0, .num_devices = 4}};

  TF_ASSERT_OK_AND_ASSIGN(auto op_sharding, LogicalToPhysicalSharding(
                                                {{}, {}, {}}, devices, {2, 2}));

  EXPECT_EQ(op_sharding.type(), OpSharding::REPLICATED);

  TF_ASSERT_OK_AND_ASSIGN(op_sharding,
                          LogicalToPhysicalSharding({}, devices, {2, 2},
                                                    /*list_all_devices=*/true));

  EXPECT_EQ(op_sharding.type(), OpSharding::REPLICATED);

  TF_ASSERT_OK_AND_ASSIGN(
      op_sharding, LogicalToPhysicalSharding({{}, {}, {}}, devices, {2, 2},
                                             /*list_all_devices=*/true));
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          HloSharding::FromProto(op_sharding));
  EXPECT_THAT(hlo_sharding.tile_assignment().array(), ElementsAre(0, 1, 2, 3));
}

TEST(MultiMeshShardingTest, ResizeSharding) {
  auto input = HloSharding::IotaTile({4, 1, 1});
  Shape shape{PrimitiveType::F32, {8, 8, 8}, {}};
  auto result = ResizeSharding(shape, input, /*num_elements=*/2);
  EXPECT_TRUE(result.has_value());

  auto bad_result = ResizeSharding(shape, input, /*num_elements=*/5);
  EXPECT_FALSE(bad_result.has_value());
}

TEST(MultiMeshShardingTest, DoNotOvershardShape) {
  auto input = HloSharding::IotaTile({2, 1, 2});
  Shape shape{PrimitiveType::F32, {2, 8, 8}, {}};
  auto result = ResizeSharding(shape, input, /*num_elements=*/16);
  ASSERT_TRUE(result.has_value());

  EXPECT_THAT(result->tile_assignment().dimensions(),
              Pointwise(Le(), shape.dimensions()));
}

TEST(MultiMeshShardingTest, BasicReplicateOnDims) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 4
    iota_reshape_dims: 8
    iota_transpose_perm: 0
  )";
  static constexpr absl::string_view kCorrectPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 2
    iota_reshape_dims: 2
    iota_reshape_dims: 4
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input, GetSharding(kInputPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding test, ReplicateDims({0}, input));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding correct, GetSharding(kCorrectPbtxt));
  EXPECT_EQ(test, correct);
};

TEST(MultiMeshShardingTest, BasicReplicateOnMultipleDims) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 3
    iota_reshape_dims: 24
    iota_transpose_perm: 0
  )";
  static constexpr absl::string_view kCorrectPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 3
    tile_assignment_dimensions: 8
    iota_reshape_dims: 8
    iota_reshape_dims: 3
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input, GetSharding(kInputPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding test, ReplicateDims({0, 1}, input));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding correct, GetSharding(kCorrectPbtxt));
  EXPECT_EQ(test, correct);
};

TEST(MultiMeshShardingTest, BasicReplicateOnMultipleNonContiguousDims) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 3
    iota_reshape_dims: 24
    iota_transpose_perm: 0
  )";
  static constexpr absl::string_view kCorrectPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 6
    iota_reshape_dims: 2
    iota_reshape_dims: 4
    iota_reshape_dims: 3
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    iota_transpose_perm: 2
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input, GetSharding(kInputPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding test, ReplicateDims({0, 2}, input));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding correct, GetSharding(kCorrectPbtxt));
  EXPECT_EQ(test, correct);
};

TEST(MultiMeshShardingTest, BasicReplicateOnDims4d) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 3
    iota_reshape_dims: 24
    iota_transpose_perm: 0
  )";
  static constexpr absl::string_view kCorrectPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 3
    tile_assignment_dimensions: 4
    iota_reshape_dims: 2
    iota_reshape_dims: 4
    iota_reshape_dims: 1
    iota_reshape_dims: 3
    iota_reshape_dims: 1
    iota_transpose_perm: 0
    iota_transpose_perm: 4
    iota_transpose_perm: 2
    iota_transpose_perm: 3
    iota_transpose_perm: 1
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input, GetSharding(kInputPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding test, ReplicateDims({1}, input));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding correct, GetSharding(kCorrectPbtxt));
  EXPECT_EQ(test, correct);
};

TEST(MultiMeshShardingTest, ReplicateOnDimsLastTileDimReplicated) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 3
    iota_reshape_dims: 2
    iota_reshape_dims: 3
    iota_reshape_dims: 4
    iota_transpose_perm: 0
    iota_transpose_perm: 2
    iota_transpose_perm: 1
    replicate_on_last_tile_dim: true
  )";
  static constexpr absl::string_view kCorrectPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 6
    iota_reshape_dims: 6
    iota_reshape_dims: 4
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input, GetSharding(kInputPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding test, ReplicateDims({1}, input));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding correct, GetSharding(kCorrectPbtxt));
  EXPECT_EQ(test, correct);
};

TEST(MultiMeshShardingTest, ReplicateDataParallelDotTransposeInput) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 8
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 1
    iota_reshape_dims: 2
    iota_reshape_dims: 8
    iota_transpose_perm: 1
    iota_transpose_perm: 0
  )";
  static constexpr absl::string_view kCorrectPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 8
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 2
    iota_reshape_dims: 2
    iota_reshape_dims: 8
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input, GetSharding(kInputPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding test, ReplicateDims({3}, input));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding correct, GetSharding(kCorrectPbtxt));
  EXPECT_EQ(test, correct);
}

TEST(MultiMeshShardingTest, ReplicateDataParallelDot) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 8
    iota_reshape_dims: 16
    iota_transpose_perm: 0
  )";
  static constexpr absl::string_view kCorrectPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 8
    tile_assignment_dimensions: 2
    iota_reshape_dims: 2
    iota_reshape_dims: 1
    iota_reshape_dims: 8
    iota_transpose_perm: 2
    iota_transpose_perm: 0
    iota_transpose_perm: 1
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input, GetSharding(kInputPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding test, ReplicateDims({0}, input));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding correct, GetSharding(kCorrectPbtxt));
  EXPECT_EQ(test, correct);
};

TEST(MultiMeshShardingTest, ReplicateOnDimsLastTileDimReplicated4d) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 3
    iota_reshape_dims: 16
    iota_reshape_dims: 3
    iota_transpose_perm: 0
    iota_transpose_perm: 1
    replicate_on_last_tile_dim: true
  )";
  static constexpr absl::string_view kCorrectPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 6
    iota_reshape_dims: 2
    iota_reshape_dims: 8
    iota_reshape_dims: 3
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    iota_transpose_perm: 2
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input, GetSharding(kInputPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding test, ReplicateDims({0}, input));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding correct, GetSharding(kCorrectPbtxt));
  EXPECT_EQ(test, correct);
};

TEST(MultiMeshShardingTest, MultipleReplicatedDimWithPermutation) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 8
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 1
    iota_reshape_dims: 2
    iota_reshape_dims: 8
    iota_transpose_perm: 1
    iota_transpose_perm: 0
  )";
  static constexpr absl::string_view kCorrectPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 8
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 2
    iota_reshape_dims: 2
    iota_reshape_dims: 8
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input, GetSharding(kInputPbtxt));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding test, ReplicateDims({3, 4}, input));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding correct, GetSharding(kCorrectPbtxt));
  EXPECT_EQ(test, correct);
};

TEST(MultiMeshShardingTest, CanonicalizeBasicSharding) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 3
    iota_reshape_dims: 3
    iota_reshape_dims: 16
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input_sharding, GetSharding(kInputPbtxt));
  static constexpr absl::string_view kInputShape = R"(
      element_type: BF16
      dimensions: 4
      dimensions: 8
      dimensions: 12
      dimensions: 4
      is_dynamic_dimension: false
      is_dynamic_dimension: false
      is_dynamic_dimension: false
      is_dynamic_dimension: false
  )";
  TF_ASSERT_OK_AND_ASSIGN(Shape input_shape, GetShape(kInputShape));
  TF_ASSERT_OK_AND_ASSIGN(auto zuku_shape,
                          CanonicalizeSharding(input_sharding, input_shape,
                                               /*shape_is_global=*/true));

  static const zuku::ShardedShape kCorrectZukuShape{
      .type = zuku::SupportedType::BF16,
      .sharding{.dims =
                    {
                        zuku::ShardingDim{
                            .size = 4,
                            .sharding = 1,
                            .permutation = 1,
                        },
                        zuku::ShardingDim{
                            .size = 8,
                            .sharding = 2,
                            .permutation = 2,
                        },
                        zuku::ShardingDim{
                            .size = 12,
                            .sharding = 4,
                            .permutation = 3,
                        },
                        zuku::ShardingDim{
                            .size = 4,
                            .sharding = 2,
                            .permutation = 4,
                        },
                        zuku::ShardingDim{
                            .size = 1,
                            .sharding = 3,
                            .permutation = 0,
                        },
                    },
                .devices = zuku::DeviceList{{.start = 0, .num_devices = 48}}},
  };

  EXPECT_EQ(zuku_shape, kCorrectZukuShape);
}

TEST(MultiMeshShardingTest, CanonicalizePermutedReplicatedSharding) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 1
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 3
    iota_reshape_dims: 2
    iota_reshape_dims: 4
    iota_reshape_dims: 2
    iota_reshape_dims: 3
    iota_transpose_perm: 2
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    iota_transpose_perm: 3
    replicate_on_last_tile_dim: true
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input_sharding, GetSharding(kInputPbtxt));
  static constexpr absl::string_view kInputShape = R"(
      element_type: BF16
      dimensions: 4
      dimensions: 4
      dimensions: 8
      dimensions: 4
      is_dynamic_dimension: false
      is_dynamic_dimension: false
      is_dynamic_dimension: false
      is_dynamic_dimension: false
  )";
  TF_ASSERT_OK_AND_ASSIGN(Shape input_shape, GetShape(kInputShape));
  TF_ASSERT_OK_AND_ASSIGN(auto zuku_shape,
                          CanonicalizeSharding(input_sharding, input_shape,
                                               /*shape_is_global=*/true));

  static const zuku::ShardedShape kCorrectZukuShape{
      .type = zuku::SupportedType::BF16,
      .sharding{.dims =
                    {
                        zuku::ShardingDim{
                            .size = 4,
                            .sharding = 1,
                            .permutation = 2,
                        },
                        zuku::ShardingDim{
                            .size = 4,
                            .sharding = 2,
                            .permutation = 3,
                        },
                        zuku::ShardingDim{
                            .size = 8,
                            .sharding = 4,
                            .permutation = 1,
                        },
                        zuku::ShardingDim{
                            .size = 4,
                            .sharding = 2,
                            .permutation = 0,
                        },
                        zuku::ShardingDim{
                            .size = 1,
                            .sharding = 3,
                            .permutation = 4,
                        },
                    },
                .devices = zuku::DeviceList{{.start = 0, .num_devices = 48}}},
  };

  EXPECT_EQ(zuku_shape, kCorrectZukuShape);
}

TEST(MultiMeshShardingTest, CanonicalizePermutedSharding) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 4
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 3
    iota_reshape_dims: 6
    iota_reshape_dims: 4
    iota_transpose_perm: 1
    iota_transpose_perm: 0
  )";
  TF_ASSERT_OK_AND_ASSIGN(HloSharding input_sharding, GetSharding(kInputPbtxt));
  static constexpr absl::string_view kInputShape = R"(
      element_type: BF16
      dimensions: 8
      dimensions: 4
      dimensions: 6
      is_dynamic_dimension: false
      is_dynamic_dimension: false
      is_dynamic_dimension: false
  )";
  TF_ASSERT_OK_AND_ASSIGN(Shape input_shape, GetShape(kInputShape));
  TF_ASSERT_OK_AND_ASSIGN(auto zuku_shape,
                          CanonicalizeSharding(input_sharding, input_shape,
                                               /*shape_is_global=*/true));

  static const zuku::ShardedShape kCorrectZukuShape{
      .type = zuku::SupportedType::BF16,
      .sharding{.dims =
                    {
                        zuku::ShardingDim{
                            .size = 8,
                            .sharding = 4,
                            .permutation = 2,
                        },
                        zuku::ShardingDim{
                            .size = 4,
                            .sharding = 2,
                            .permutation = 0,
                        },
                        zuku::ShardingDim{
                            .size = 6,
                            .sharding = 3,
                            .permutation = 1,
                        },
                    },
                .devices = zuku::DeviceList{{.start = 0, .num_devices = 24}}},
  };

  EXPECT_EQ(zuku_shape, kCorrectZukuShape);
}

TEST(MultiMeshShardingTest, PermutedIotaInSingleDimSharding) {
  static constexpr absl::string_view kInputPbtxt = R"(
    type: OTHER
    tile_assignment_dimensions: 8
    tile_assignment_dimensions: 1
    iota_reshape_dims: 2
    iota_reshape_dims: 2
    iota_reshape_dims: 2
    iota_transpose_perm: 1
    iota_transpose_perm: 0
    iota_transpose_perm: 2
  )";

  TF_ASSERT_OK_AND_ASSIGN(HloSharding input_sharding, GetSharding(kInputPbtxt));
  Shape shape{PrimitiveType::F32, {8, 8}, {}};
  // we cannot do an iota permutation with a single tensor dimension
  auto status_or = CanonicalizeSharding(input_sharding, shape, true);
  EXPECT_FALSE(status_or.status().ok());
}

}  // namespace
}  // namespace xla
