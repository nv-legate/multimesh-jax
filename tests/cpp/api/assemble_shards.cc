/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "legate_xla_utils.h"
#include "gmock/gmock.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <legate_xla_common.h>
#include <numeric>
#include <xla_to_legate.h>

using ::testing::ElementsAreArray;

MATCHER_P(ShardIsVector, values, "") { return arg == values; }

class IndexIterator {
public:
  IndexIterator(const legate_xla::Shape &shape) : index_(shape.dims.size(), 0) {
    if (shape.tile_shape.has_value()) {
      shape_.resize(shape.dims.size());
      for (size_t dim = 0; dim < shape.dims.size(); ++dim) {
        shape_[dim] = shape.dims[dim] / (*shape.tile_shape)[dim];
      }
    } else {
      // no tiling, color shape is all ones
      shape_ = std::vector<int64_t>(shape.dims.size(), 1);
    }
  }

  const std::vector<int64_t> &index() const { return index_; }

  void operator++() {
    size_t dim = 0;
    for (; dim < shape_.size(); ++dim) {
      index_[dim]++;
      if (index_[dim] < shape_[dim]) {
        break;
      }
      // roll over
      index_[dim] = 0;
    }
    if (dim == shape_.size()) {
      done_ = true;
    }
  }

  bool done() const { return done_; }

public:
  bool done_{};
  std::vector<int64_t> index_{};
  std::vector<int64_t> shape_{};
};

void test_legate_shape(legate_xla::Shape shape) {
  size_t shard_num_elements = 1;
  const auto &tile =
      shape.tile_shape.has_value() ? *shape.tile_shape : shape.dims;
  for (size_t dim : tile) {
    shard_num_elements *= dim;
  }
  size_t shard_size = shard_num_elements * sizeof(int32_t);

  size_t total_elements = 1;
  for (size_t dim : shape.dims) {
    total_elements *= dim;
  }

  size_t num_shards = total_elements / shard_num_elements;
  std::vector<std::vector<int32_t>> input_shards(num_shards);

  for (auto &shard : input_shards) {
    shard.resize(shard_num_elements);
    std::iota(shard.begin(), shard.end(), 0);
  }

  std::vector<legate_xla::Shard> shards;
  shards.reserve(num_shards);

  IndexIterator it(shape);
  for (int shard = 0; shard < num_shards; ++shard, ++it) {
    int *ptr;
    auto result = cudaMalloc(&ptr, shard_size);
    ASSERT_EQ(result, cudaSuccess);
    cudaMemcpy(ptr, input_shards[shard].data(), shard_size,
               cudaMemcpyHostToDevice);
    shards.push_back(legate_xla::Shard{.data = ptr,
                                       .local_device_id = shard,
                                       .shape_index = it.index(),
                                       .size = shard_size});
  }

  auto store = legate_xla::AssembleShards(shape, shards, {0, num_shards});

  std::vector<void *> device_slices(num_shards);
  legate_xla::SliceLocalShards(store, device_slices, {0, num_shards});

  std::vector<std::vector<int32_t>> slices;
  slices.reserve(num_shards);
  for (void *device_buf : device_slices) {
    std::vector<int32_t> slice(shard_num_elements);
    cudaMemcpy(slice.data(), device_buf, shard_size, cudaMemcpyDeviceToHost);
    slices.push_back(std::move(slice));
  }

  EXPECT_THAT(slices, ElementsAreArray(input_shards));
}

void test_shape(std::vector<int64_t> dims) {
  int64_t num_devices = legate_xla::GetLocalDevices(0).size();
  int64_t replication = 1;
  std::vector<int64_t> tile_shape;
  if (dims.size() > 0 && dims[0] >= num_devices &&
      (dims[0] % num_devices == 0)) {
    tile_shape = dims;
    tile_shape[0] = dims[0] / num_devices;
  } else if (dims.empty()) {
    replication = num_devices;
  } else {
    num_devices = 1;
  }

  test_legate_shape({.type = legate_xla::SupportedType::S32,
                     .dims = dims,
                     .tile_shape = tile_shape,
                     .replicated = replication});
}

TEST(AssembleShardsTest, BasicAssemble) {
  test_shape({8});
  test_shape({4, 4});
  test_shape({2, 2, 2});
  test_shape({2, 1, 7});
}
