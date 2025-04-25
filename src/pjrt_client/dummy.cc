/* clang-format off
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// This silences the linker error by providing the symbol
extern "C" void* PyInit_liblegate_xla_client() { return nullptr; }
