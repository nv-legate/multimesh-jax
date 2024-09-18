#! /usr/bin/env bash

pushd /opt/xla

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

BAZEL_CACHE=grpc://host.docker.internal:9092

conda run --no-capture-out -n legere python configure.py \
  --backend CUDA \
  --host_compiler GCC \
  --nccl \
  --nccl_version=$TF_NCCL_VERSION \
  --cuda_compute_capabilities=sm_80,sm_90a

python_version=$(/opt/install/miniconda/envs/legere/bin/python --version | awk '{print $2}' | cut -d . -f 1-2)
echo "build --repo_env HERMETIC_PYTHON_VERSION=${python_version}" >> xla_configure.bazelrc

popd

pushd /opt/legate-jax

conda run --no-capture-out -n legere cmake -S . -B build -DCMAKE_GENERATOR:STRING="Unix Makefiles" \
  -DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ -DCMAKE_C_COMPILER:PATH=/usr/bin/gcc \
  -DCMAKE_BUILD_TYPE:STRING="${LEGATE_BUILD_TYPE}" \
  -DCMAKE_LIBRARY_PATH:STRING=/usr/lib/x86_64-linux-gnu -DCMAKE_CXX_STANDARD:STRING=17 \
  -DLegateXLA_BAZEL_REMOTE_CACHE:STRING="${BAZEL_CACHE}" \
  -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release \
  -DCPM_xla_SOURCE:PATH=/opt/xla -Dlegate_core_ROOT:PATH=/opt/legate/build \
  -DLegateXLA_ASAN:BOOL=OFF \
  -DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
  -DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere

conda run --no-capture-out -n legere cmake --build build --parallel
