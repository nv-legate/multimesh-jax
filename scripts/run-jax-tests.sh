#! /usr/bin/env bash

jax_dir=`python -c 'import jax; from pathlib import Path; print(Path(jax.__file__).parent.parent)'`
echo $jax_dir

num_gpus=${1:-1}

export LEGION_DEFAULT_ARGS="-ll:py 0 \
 -lg:local 0 \
 -ll:cpu 4 \
 -ll:gpu $num_gpus \
 -cuda:skipbusy \
 -ll:util 2 \
 -ll:csize 4000 \
 -ll:fsize 4000 \
 -ll:zsize 32 \
 -lg:eager_alloc_percentage 50"

export JAX_PLATFORMS=legate,cuda

function run_jax_test() {
  test=$1
  python $jax_dir/tests/$test.py
}

run_jax_test pjit_test

