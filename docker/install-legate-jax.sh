#! /usr/bin/env bash

editable_flag=${1:-}

pushd /opt/legate-jax
export CMAKE_ARGS="-DCPM_xla_SOURCE:PATH=/opt/xla -DLegateJAX_ROOT:PATH=/opt/legate-jax/build"
export SETUPTOOLS_ENABLE_FEATURES=legacy-editable
conda run --no-capture-out -n legere cmake --install build --prefix /opt/install/miniconda/envs/legere
conda run --no-capture-out -n legere python -m pip install $editable_flag .
