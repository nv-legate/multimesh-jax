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

#include <gtest/gtest.h>
#include <legate_xla_common.h>
#include <numeric>
#include <xla_to_legate.h>
#include <zuku/shape.h>

using SupportedType = zuku::SupportedType;
static constexpr std::array all_types = {
    SupportedType::S32, SupportedType::F32, SupportedType::F64,
    SupportedType::S8,  SupportedType::S16, SupportedType::S32,
    SupportedType::S64, SupportedType::U8,  SupportedType::U16,
    SupportedType::U32, SupportedType::U64};

class TestBufferAction : public legate_xla::BufferAction {
public:
  explicit TestBufferAction(std::function<void(void *, int)> action)
      : action_(std::move(action)){};

  void Act(void *dst, int local_device_id) override {
    action_(dst, local_device_id);
  }

private:
  std::function<void(void *, int)> action_;
};

template <class T>
void test_create_store_tmpl(zuku::ShardedShape shape, size_t num_devices) {
  auto type_size = zuku::SupportedTypeSizeOf(shape.type);
  auto num_elements = legate_xla::ShapeNumElements(shape);
  size_t shard_num_elements = num_elements / num_devices;
  size_t shard_size = shard_num_elements * type_size;

  std::vector<T> elements(num_elements);
  std::iota(elements.begin(), elements.end(), 0);

  auto store = legate_xla::CreateStore(shape);

  TestBufferAction action([=](void *dst, int device) {
    const T *src = &elements[shard_num_elements * device];
    cudaMemcpy(dst, src, shard_size, cudaMemcpyHostToDevice);
  });
  std::vector<legate_xla::BufferAction *> actions;
  actions.reserve(num_devices);
  for (int dev = 0; dev < num_devices; ++dev) {
    actions.push_back(&action);
  }

  legate_xla::StoreBufferAction(
      actions, store, {.blocking = true, .machine_slice = {0, num_devices}});

  std::vector<void *> local_shards(num_devices);
  legate_xla::SliceLocalShards(store, local_shards, {0, num_devices});

  std::vector<T> final_elements(num_elements);
  size_t offset = 0;
  for (const void *device_src : local_shards) {
    cudaMemcpy(&final_elements[offset], device_src, shard_size,
               cudaMemcpyDeviceToHost);
    offset += shard_num_elements;
  }

  EXPECT_EQ(elements, final_elements);

  // destroy store
  legate_xla::Destroy(store);
}

void test_create_store(legate_xla::Shape shape, size_t num_devices) {
  switch (shape.type) {
  case SupportedType::F32:
    return test_create_store_tmpl<float>(shape, num_devices);
  case SupportedType::F64:
    return test_create_store_tmpl<double>(shape, num_devices);
  case SupportedType::S8:
    return test_create_store_tmpl<int8_t>(shape, num_devices);
  case SupportedType::S16:
    return test_create_store_tmpl<int16_t>(shape, num_devices);
  case SupportedType::S32:
    return test_create_store_tmpl<int32_t>(shape, num_devices);
  case SupportedType::S64:
    return test_create_store_tmpl<int64_t>(shape, num_devices);
  case SupportedType::U8:
    return test_create_store_tmpl<uint8_t>(shape, num_devices);
  case SupportedType::U16:
    return test_create_store_tmpl<uint16_t>(shape, num_devices);
  case SupportedType::U32:
    return test_create_store_tmpl<uint32_t>(shape, num_devices);
  case SupportedType::U64:
    return test_create_store_tmpl<uint64_t>(shape, num_devices);
  default:
    std::cerr << (int)shape.type << std::endl;
    throw std::runtime_error("unsupported test type");
  };
}

void test_all_types(std::vector<int64_t> dims) {
  int64_t num_devices = legate_xla::GetLocalDevices().size();
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

  std::optional<std::vector<int64_t>> tile;
  if (num_devices > 1) {
    tile = std::move(tile_shape);
  }

  for (auto type : all_types) {
    legate_xla::Shape shape = {.type = type,
                               .dims = dims,
                               .tile_shape = tile,
                               .explicit_replication = replication};
    test_create_store(shape, num_devices);
  }
}

TEST(CreateStore, StandardStorages) {
  test_all_types({2});
  test_all_types({4});
  test_all_types({4, 2});
  test_all_types({6, 7});
  test_all_types({4, 2, 3});
  test_all_types({2, 4, 1});
  test_all_types({2, 2, 3, 8});
  test_all_types({4, 2, 3, 4});
  test_all_types({2, 2, 4, 4});
  test_all_types({2, 1, 3, 1});
}

TEST(CreateStore, ScalarStorages) {
  std::vector<int64_t> dims;
  test_all_types(dims);
}

TEST(CreateStore, EmptyStorages) {
  test_all_types({0, 2, 3, 4});
  test_all_types({1, 0, 3, 4});
  test_all_types({1, 2, 0, 4});
  test_all_types({1, 2, 3, 0});
}
