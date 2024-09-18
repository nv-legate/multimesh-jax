#! /usr/bin/env bash

editable_flag=${1:-}

export NVTE_FRAMEWORK=jax

pushd /opt/te
conda run -n legere python -m pip install $editable_flag . --no-build-isolation

rm -rf _skbuild
