#! /usr/bin/env bash


pushd /opt/legate
export CMAKE_ARGS="-DFIND_LEGATE_CORE_CPP:BOOL=ON -Dlegate_core_ROOT:PATH=/opt/legate/build"

conda run --no-capture-out -n legere cmake --install build --prefix /opt/install/miniconda/envs/legere
