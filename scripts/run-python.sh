#! /usr/bin/env bash
jax_dir=`python -c 'import jax; from pathlib import Path; print(Path(jax.__file__).parent.parent)'`
echo $jax_dir

VALID_ARGS=$(getopt -o ac:d:f:gmn:p:t:x --long asan,cpus:,dump,debug:,filter:,gdb,gpus:,metadata-off,mpi:,test: -- "$@")
if [[ $? -ne 0 ]]; then
    exit 1;
fi

gpus=`nvidia-smi --list-gpus | wc -l`
cpus=4
debug=0
launcher=""
include_metadata=false
network=none

eval set -- "$VALID_ARGS"
while [ : ]; do
  case "$1" in
    -a | --asan)
        echo "Running with address sanitizer"
        export ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=0:halt_on_error=1:new_delete_type_mismatch=0
        export LD_PRELOAD="$(gcc -print-file-name=libasan.so) $(gcc -print-file-name=libstdc++.so)"
        shift
        ;;
    -c | --cpus)
        echo "Running $2 CPUS"
        cpus=$2
        export XLA_FLAGS="${XLA_FLAGS} --xla_force_host_platform_device_count=$2"
        shift 2
        ;;
    -d | --debug)
        echo "Running with debug=$2"
        debug=$2
        shift 2
        ;;
    -g | --gdb)
        echo "Running with gdb"
        launcher="gdb --args"
        shift
        ;;
    -p | --mpi)
        echo "Running with $2 MPI procs"
        launcher="mpirun -n $2"
        network=mpi
        shift 2
        ;;
    -n | --gpus)
        gpus=$2
        shift 2
        ;;
    -x | --dump)
        echo "Dumping XLA output"
        export XLA_FLAGS="${XLA_FLAGS} --xla_dump_to=dump --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_hlo_pass_re=.*"
        shift
        ;;
    -o | --metadata-off)
        include_metadata=true
        export XLA_FLAGS="${XLA_FLAGS} --xla_dump_module_metadata=false --xla_dump_disable_metadata"
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
 -ll:networks $network \
 -ll:ib_rsize 0 \
 -logfile test.log \
 $level \
 -lg:eager_alloc_percentage 50"

export JAX_PLATFORMS=legate,cuda,cpu
export TF_CPP_MIN_LOG_LEVEL=$min_level
export TF_CPP_MAX_LOG_LEVEL=$debug
export TF_CPP_VMODULE=legate_pjrt_buffer=$debug,legate_computation=$debug,legate_pjrt_client=$debug,hlo_partition=$debug,legate_pjrt_executable=$debug,loop_scheduler=$debug,legate_ifrt_client=$debug,pjrt_executable=$debug,loop_scheduler=$debug,topology_util=$debug,legate_utils=$debug,platform_util=$debug
export JAX_TRACEBACK_FILTERING=off
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=0
export UCX_TLS=^mm

echo "Running $@ with ${gpus} GPUS"

$launcher python $@

