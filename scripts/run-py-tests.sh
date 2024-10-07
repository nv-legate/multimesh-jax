SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e

python $SCRIPT_DIR/../tests/python/jax_microbatch_test.py --gpus 1 $@
python $SCRIPT_DIR/../tests/python/jax_microbatch_test.py --cpus 4 --gpus 0 $@
python $SCRIPT_DIR/../tests/python/jax_task_test.py       --gpus 1 $@
python $SCRIPT_DIR/../tests/python/jax_task_test.py       --gpus 2 $@
python $SCRIPT_DIR/../tests/python/jax_auto_task_test.py  --gpus 2 $@
python $SCRIPT_DIR/../tests/python/jax_auto_task_test.py  --cpus 4 --gpus 0 $@
python $SCRIPT_DIR/../tests/python/jax_auto_task_test.py  --cpus 8 --gpus 0 $@

