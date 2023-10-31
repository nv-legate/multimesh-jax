#! /usr/bin/env bash

# the path of Python test to run
test_path=$1

# the degree of debug output (VLOG)
debug=${2:-0}

# optional: an individual test to run
test=${3:-}

# no. gpus
system_gpus=`nvidia-smi --list-gpus | wc -l`
gpus=${4:-${system_gpus}}

# env variable for JAX_PLATFORMS
platform=legate

if [ ! -z $test ]; then
  test_flag="--test_targets=${test}"
fi

if [ $debug -eq "0" ]; then
  min_level=3
  thunk_debug=0
  level=""
else
  min_level=0
  thunk_debug=100
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



TF_CPP_MIN_LOG_LEVEL=$min_level \
TF_CPP_MAX_LOG_LEVEL=$debug \
JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=0 \
JAX_TRACEBACK_FILTERING=off \
TF_CPP_VMODULE=legate_ifrt_client=$debug,legate_pjrt_client=$debug,pjrt_client=$debug,tfrt_cpu_pjrt_client=$debug,hlo_partition=$debug,legate_computation=$debug \
XLA_FLAGS="--xla_dump_to=dump_$1 --xla_dump_hlo_as_text --xla_gpu_enable_xla_runtime_executable" \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
JAX_PLATFORMS=$platform,cuda \
python $test_path $test_flag

