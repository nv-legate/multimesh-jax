/* Copyright 2022 NVIDIA Corporation
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

#pragma once

#include "legate.h"
#include "legate_xla_c.h"

namespace legate_xla {

extern Legion::Logger log_xla;
struct Registry {
  static legate::TaskRegistrar &get_registrar();
};

template <typename T> struct XlaTask : public legate::LegateTask<T> {
  using Registrar = Registry;
};

// Generic initialization of LegateBuffer from host data ptr
class XLAInitFromHostTask : public XlaTask<XLAInitFromHostTask> {
public:
  static const int32_t TASK_ID = XlaOpCode::XLA_INIT_FROM_HOST_TASK;

public:
  static void gpu_variant(legate::TaskContext context);
};

// Generic initialization of LegateBuffer to zero
class XLAInitZeroTask : public XlaTask<XLAInitZeroTask> {
public:
  static const int32_t TASK_ID = XlaOpCode::XLA_INIT_ZERO_TASK;

public:
  static void gpu_variant(legate::TaskContext context);
};

} // namespace legate_xla
