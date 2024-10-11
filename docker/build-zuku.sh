#! /usr/bin/env bash

pushd /opt/zuku

conda run --no-capture-out -n legere \
  cmake -S . -B build -DCMAKE_GENERATOR:STRING=Ninja \
  -DCMAKE_CXX_COMPILER:PATH=/usr/bin/g++ -DCMAKE_C_COMPILER:PATH=/usr/bin/gcc \
  -DCMAKE_LIBRARY_PATH:STRING=/usr/lib/x86_64-linux-gnu -DCMAKE_CXX_STANDARD:STRING=17 \
  -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING="${LEGATE_BUILD_TYPE}" \
  -DLegion_ROOT:PATH=/opt/install/miniconda/envs/legere \
  -DCMAKE_INSTALL_RPATH:PATH=/opt/install/miniconda/envs/legere/lib \
  -DCMAKE_INSTALL_PREFIX:PATH=/opt/install/miniconda/envs/legere
conda run --no-capture-out -n legere cmake --build build

cmake --build build

popd
