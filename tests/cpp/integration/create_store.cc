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

#include "../../src/legate_xla_utils.h"
#include <gtest/gtest.h>
#include <legate_xla_common.h>
#include <numeric>
#include <xla_to_legate.h>

using SupportedType = legate_xla::SupportedType;
const std::array<SupportedType, 15> all_types = {
    SupportedType::PRED, SupportedType::F16, SupportedType::BF16,
    SupportedType::F32,  SupportedType::F64, SupportedType::S8,
    SupportedType::S16,  SupportedType::S32, SupportedType::S64,
    SupportedType::U8,   SupportedType::U16, SupportedType::U32,
    SupportedType::U64,  SupportedType::C64, SupportedType::C128};

void test_create_store(legate_xla::Shape shape) {

  auto type_size = legate_xla::SupportedTypeSizeOf(shape.type);
  auto elements = legate_xla::ShapeNumElements(shape);
  auto total_bytes = type_size * elements;

  std::vector<std::byte> host_data(total_bytes + 3);
  // some initialization based on int values
  {
    int32_t *host_data_as_int = (int32_t *)host_data.data();
    std::iota(host_data_as_int,
              host_data_as_int + ((int)((total_bytes + 3) / 4)), 0);
  }

  // create & fill store
  auto my_store = legate_xla::CreateStore(shape);
  std::function<void()> on_done_with_host_buffer = nullptr;
  legate_xla::CreateStoreFromHostBufferTask(
      host_data.data(), total_bytes, my_store,
      std::move(on_done_with_host_buffer));

  // check resulting data
  std::vector<std::byte> host_target_data(total_bytes);
  {
    std::function<void(const void *)> copy_func =
        [target_bytes = total_bytes, target = host_target_data.data()](
            const void *source) { memcpy(target, source, target_bytes); };
    legate_xla::CopyStoreToHostSync(my_store, copy_func);

    for (int i = 0; i < total_bytes; ++i)
      EXPECT_EQ(host_target_data[i], host_data[i]);
  }

  // destroy store
  legate_xla::Destroy(my_store);
}

void test_all_types(std::vector<size_t> dims) {
  for (auto type : all_types) {
    legate_xla::Shape shape = {.type = type, .dims = dims};
    test_create_store(shape);
  }
}

TEST(CreateStore, StandardStorages) {
  test_all_types({1});
  test_all_types({3});
  test_all_types({4, 2});
  test_all_types({3, 7});
  test_all_types({4, 2, 3});
  test_all_types({1, 4, 1});
  test_all_types({1, 2, 3, 7});
  test_all_types({4, 2, 3, 4});
  test_all_types({1, 2, 4, 4});
  test_all_types({1, 2, 3, 1});
}

TEST(CreateStore, ScalarStorages) {
  std::vector<size_t> dims;
  test_all_types(dims);
}

TEST(CreateStore, EmptyStorages) {
  test_all_types({0, 2, 3, 4});
  test_all_types({1, 0, 3, 4});
  test_all_types({1, 2, 0, 4});
  test_all_types({1, 2, 3, 0});
}

TEST(CreateStore, LargeDimensionStorages) {
  test_all_types({1, 2, 3, 4, 5, 6, 7});
  test_all_types({1, 2, 2, 1, 3});
  test_all_types({4, 2, 7, 1, 1, 2, 5, 0});
  test_all_types({1, 2, 0, 0, 0, 3, 7});
  test_all_types({4, 2, 3, 4, 0});
}
