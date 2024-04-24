#! /usr/bin/env bash

BAZEL_CACHE=$1

export PYTHON_BIN_PATH=/opt/install/miniconda/envs/legere/bin/python
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_CUDA=1
export TF_CUDA_CLANG=0
export TF_CUDA_COMPUTE_CAPABILITIES=sm_80,sm_90a
export GCC_HOST_COMPILER_PATH=/usr/bin/gcc
export TF_SET_ANDROID_WORKSPACE=0
export TF_NEED_CUDA=1
export TF_NEED_CUTENSOR=1
export TF_NEED_TENSORRT=0
export TF_CUDA_PATHS=/usr,/usr/local/cuda
export TF_CUDNN_PATHS=/usr/lib/$(uname -p)-linux-gnu
export TF_CUDA_VERSION=$(ls /usr/local/cuda/lib64/libcudart.so.*.*.* | cut -d . -f 3-4)
export TF_CUBLAS_VERSION=$(ls /usr/local/cuda/lib64/libcublas.so.*.*.* | cut -d . -f 3)
export TF_CUDNN_VERSION=$(echo "${NV_CUDNN_VERSION}" | cut -d . -f 1)
export TF_NCCL_VERSION=$(echo "${NCCL_VERSION}" | cut -d . -f 1)

conda run --no-capture-out -n legere python build/build.py \
  --bazel_options=--override_repository=xla=/opt/xla --bazel_startup_options=--batch \
  --bazel_options=--remote_cache=${BAZEL_CACHE} \
  --cuda_path=$TF_CUDA_PATHS \
  --cudnn_path=$TF_CUDNN_PATHS \
  --cuda_version=$TF_CUDA_VERSION \
  --cudnn_version=$TF_CUDNN_VERSION \
  --cuda_compute_capabilities=$TF_CUDA_COMPUTE_CAPABILITIES \
  --enable_cuda=true \
  --enable_nccl=true
