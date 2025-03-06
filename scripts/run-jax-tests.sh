#! /usr/bin/env bash
jax_dir=`python -c 'import jax; from pathlib import Path; print(Path(jax.__file__).parent.parent)'`
echo $jax_dir

VALID_ARGS=$(getopt -o ac:d:f:gn:o:t:x --long asan,dump,debug:,filter:,gdb,cpus:,gpus:,out:,test: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

gpus=`nvidia-smi --list-gpus | wc -l`
cpus=4
test=lax_numpy_test
debug=0
launcher=""
output=""

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -a | --asan)
        echo "Running with address sanitizer"
        export ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:halt_on_error=1:new_delete_type_mismatch=0
        export LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)"
        shift
        ;;
    -d | --debug)
        echo "Running with debug=$2"
        debug=$2
        shift 2
        ;;
    -c | --cpus)
        cpus=$2
        export XLA_FLAGS="${XLA_FLAGS} --xla_force_host_platform_device_count=$2"
        shift 2
        ;;
    -n | --gpus)
        gpus=$2
        shift 2
        ;;
    -g | --gdb)
        launcher="gdb --args"
        shift
        ;;
    -f | --filter)
        echo "Running filtered tests $2"
        filter=$2
        shift 2
        ;;
    -o | --output)
        echo "Piping output to $2"
        output=$2
        shift 2
        ;;
    -t | --test)
        test=$2
        shift 2
        ;;
    -x | --dump)
        echo "Dumping XLA output"
        export XLA_FLAGS="--xla_dump_to=dump --xla_dump_hlo_as_text"
        shift
        ;;
    --) shift;
        break
        ;;
  esac
done

export OMPI_MCA_plm=isolated


if [ $debug -eq "0" ]; then
  min_level=3
  level=""
else
  min_level=0
  level="-level legate.xla=1"
fi

export LEGION_DEFAULT_ARGS="-ll:py 0 \
 -lg:local 0 \
 -ll:cpu $cpus \
 -ll:gpu $gpus \
 -cuda:skipbusy \
 -ll:util 2 \
 -ll:csize 4000 \
 -ll:fsize 4000 \
 -ll:zsize 32 \
 -ll:networks none \
 -ll:ib_rsize 0 \
 $level \
 -lg:eager_alloc_percentage 50"

export JAX_PLATFORMS=legate,cuda
export TF_CPP_MIN_LOG_LEVEL=$min_level
export TF_CPP_MAX_LOG_LEVEL=$debug
#export TF_CPP_MAX_VLOG_LEVEL=$debug
export TF_CPP_VMODULE=legate_pjrt_buffer=$debug,legate_computation=$debug,legate_pjrt_client=$debug,hlo_partition=$debug,legate_pjrt_executable=$debug,legate_ifrt_client=$debug
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=0

echo "Running test ${test}"
echo "Running with ${gpus} GPUS"

if [ ! -z $filter ]; then
  test_flag="--test_targets=${filter}"
fi

if [ -z $output ]; then
  if [ -z $filter ]; then
    output=$test.out
  else
    output=$test.$filter.out
  fi
fi
$launcher python $jax_dir/tests/$test.py $test_flag 2>&1 | tee $output

