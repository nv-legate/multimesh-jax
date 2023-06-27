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
#include <thread>
#include <xla_to_legate.h>

void put_get(int thread_id, size_t test_size, size_t repetitions) {
  std::vector<int> host_data(test_size + repetitions);
  std::iota(host_data.begin(), host_data.begin() + test_size + repetitions, 0);

  std::vector<int> host_target_data(test_size);

  // setup 1d shape [test_size]
  std::vector<size_t> dims;
  dims.push_back(test_size);
  legate_xla::Shape my_1d_shape = {.type = legate_xla::SupportedType::S32,
                                   .dims = dims};
  auto my_1d_store = legate_xla::CreateStore(my_1d_shape);

  std::function<void()> on_done_with_host_buffer = nullptr;
  std::function<void(const void *)> copy_func =
      [target_bytes = test_size * sizeof(int),
       target = host_target_data.data()](const void *source) {
        memcpy(target, source, target_bytes);
      };

  for (int r = 0; r < repetitions; ++r) {
    // fill my_1d_store with host data
    legate_xla::CreateStoreFromHostBufferTask(
        host_data.data() + r, test_size * sizeof(int), my_1d_store,
        std::move(on_done_with_host_buffer));
    // fetch data from device
    CopyStoreToHostSync(my_1d_store, copy_func);

    for (int i = 0; i < test_size; ++i)
      EXPECT_EQ(host_target_data[i], host_data[i + r]);
  }
}

TEST(PutGet, SingleThreadPutGet) { put_get(0, 100, 100); }

TEST(PutGet, IssueConcurrentPutGet) {
  size_t num_threads = 2;
  size_t test_size = 100;
  size_t num_repetitions = 100;
  std::vector<std::thread> threads(num_threads);
  for (int i = 0; i < num_threads; i++) {
    threads[i] = std::thread(put_get, i, test_size, num_repetitions);
  }

  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }
}
