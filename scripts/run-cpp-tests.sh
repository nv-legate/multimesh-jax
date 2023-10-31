#! /usr/bin/env bash

# the path of Python test to run
test_path=$1

# the degree of debug output (VLOG)
debug=${2:-0}

# no. gpus
system_gpus=`nvidia-smi --list-gpus | wc -l`
gpus=${4:-${system_gpus}}

if [ ! -z $test ]; then
  test_flag="--gtest_filter=*${test}*"
fi

if [ $debug -eq "0" ]; then
  level=""
else
  level="-level legate.xla=1"
fi


export LEGION_DEFAULT_ARGS="-ll:py 0 \
 -lg:local 0 \
 -ll:cpu 4 \
 -ll:gpu ${gpus} \
 -cuda:skipbusy \
 -ll:util 2 \
 -ll:csize 4000 \
 -ll:fsize 4000 \
 -ll:zsize 32 \
 ${level} \
 -lg:eager_alloc_percentage 50"

ctest --extra-verbose --test-dir $test_path

