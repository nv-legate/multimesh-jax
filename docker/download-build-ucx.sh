#! /usr/bin/env bash

pushd opt
git clone -b master --filter=tree:0 https://github.com/openucx/ucx.git ucx

pushd ucx
./autogen.sh
./contrib/configure-release --enable-mt \
  --with-cuda=/usr/local/cuda --with-java=no

make -j
