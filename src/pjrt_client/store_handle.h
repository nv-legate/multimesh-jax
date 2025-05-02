/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _XLA_PJRT_MULTIMESH_STORE_HANDLE_H_
#define _XLA_PJRT_MULTIMESH_STORE_HANDLE_H_

#include "src/zuku/tiled_array.h"
#include "xla/pjrt/multimesh/store_handle_fwd.h"

namespace xla {

struct StoreHandleImpl;

struct StoreHandleImpl {
  zuku::Store<zuku::ShardedArray> array;
  std::string name;
  ~StoreHandleImpl();
};

}  // namespace xla

#endif
