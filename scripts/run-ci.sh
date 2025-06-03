#! /usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e

available_procs=`egrep '^core id' /proc/cpuinfo | sort -u | wc -l`
build_dir=${1:-/opt/build/multimesh-jax}
xla_path=${2:-/opt/workspace/xla}
nproc=${3:-$available_procs}

$SCRIPT_DIR/clang-tidy.sh $xla_path $nproc

CUDA_VISIBLE_DEVICES=0 ctest --test-dir $build_dir --parallel $nproc

$SCRIPT_DIR/run-py-tests.sh
