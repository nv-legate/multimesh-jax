#! /usr/bin/env bash

export PYTHON_BIN_PATH=/opt/install/miniconda/envs/legere/bin/python
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_ROCM=0
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_NCCL_VERSION=$(ls /lib/x86_64-linux-gnu/libnccl.so.*.*.* | cut -d . -f 3-5)
export TF_CUDA_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*.*.* | cut -d . -f 3-4)
export TF_CUDA_CLANG=0
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export CC_OPT_FLAGS=--Wno-sign-compare
export TF_SET_ANDROID_WORKSPACE=0

conda run --no-capture-out -n legere python configure.py \
  --backend CUDA --host_compiler GCC \
  --nccl \
  --nccl_version=$TF_NCCL_VERSION \
  --cuda_compute_capabilities=sm_80,sm_90a
