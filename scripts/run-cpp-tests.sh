#! /usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
VALID_ARGS=$(getopt -o ab:d:f:gn:o:x --long asan,build-dir:,dump,debug:,filter:,gdb,gpus:,out: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

debug=0
gpus=`nvidia-smi --list-gpus | wc -l`
# the default build directory with the tests is the
# $(top_source_dir)/build
build_dir="$SCRIPT_DIR/../build"
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
    -b | --build-dir)
        build_dir=$2
        shift 2
        ;;
    -d | --debug)
        echo "Running with debug=$2"
        debug=$2
        shift 2
        ;;
    -n | --gpus)
        gpus=$2
        shift 2
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

if [ $debug -eq "0" ]; then
  min_level=3
  level=""
else
  min_level=0
  level="-level legate.xla=1"
fi


if [ ! -z $filter ]; then
  filter="-R ${filter}"
fi

if [ $debug -eq "0" ]; then
  level=""
else
  level="-level legate.xla=1"
fi

export OMPI_MCA_plm=isolated
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
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=$min_level
export TF_CPP_MAX_LOG_LEVEL=$debug
export TF_CPP_VMODULE=legate_pjrt_buffer=$debug,legate_computation=$debug,legate_pjrt_client=$debug,hlo_partition=$debug,legate_pjrt_executable=$debug

echo "Running with ${gpus} GPUS"
echo "Running tests from directory ${build_dir}"
if [ ! -z "$filter" ]; then
  echo "Running with filter ${filter}"
fi

if [ -z $output ]; then
  if [ -z "$filter" ]; then
    output=test.out
  else
    output=test.$filter.out
  fi
fi

ctest --extra-verbose --test-dir $build_dir $filter 2>&1 | tee $output

