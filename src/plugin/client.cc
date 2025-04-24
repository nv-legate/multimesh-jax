/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "zuku/init.h"

extern "C" {

struct PJRT_Api;

PJRT_Api* GetLegatePjrtApi();

const PJRT_Api* GetPjrtApi() { return GetLegatePjrtApi(); }
}
