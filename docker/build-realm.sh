#! /usr/bin/env bash

LEGATE_BUILD_TYPE=${1:-Release}

set -e

pushd /opt/realm

conda run --no-capture-out -n legere cmake -S . -B build -DCMAKE_GENERATOR:STRING=Ninja \
  -DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ -DCMAKE_C_COMPILER:PATH=/usr/bin/gcc \
  -DCMAKE_LIBRARY_PATH:STRING=/usr/lib/x86_64-linux-gnu -DCMAKE_CXX_STANDARD:STRING=17 \
  -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING="${LEGATE_BUILD_TYPE}" -DLegion_USE_CUDA:BOOL=ON \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -ggdb" \
  -DCMAKE_CXX_FLAGS_DEBUG="-O0 -ggdb" \
  -DLegion_MAX_DIM:STRING=5 -DLegion_USE_OpenMP:BOOL=ON -DLegion_USE_GASNet:BOOL=OFF \
  -DLegion_USE_Python:BOOL=ON -DLegion_VERSION:STRING=24.9.0 -DLegion_BUILD_BINDINGS:BOOL=ON \
  -DLegion_REDOP_COMPLEX:BOOL=ON -DLegion_HIJACK_CUDART:BOOL=OFF -DLegion_GPU_REDUCTIONS:BOOL=OFF \
  -DLegion_REDOP_HALF:BOOL=ON -DCUDA_NVCC_FLAGS:STRING=-std=c++17 \
  -DLegion_MAX_NUM_NODES=4096 \
  -DLegion_DEFAULT_LOCAL_FIELDS:STRING=0 -DLegion_NETWORKS:STRING=ucx \
  -DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
  -DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere

conda run --no-capture-out -n legere cmake --build build

popd
