SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e

$SCRIPT_DIR/run-python.sh $SCRIPT_DIR/../tests/python/jax_microbatch_test.py --gpus 1
$SCRIPT_DIR/run-python.sh $SCRIPT_DIR/../tests/python/jax_task_test.py       --gpus 1
$SCRIPT_DIR/run-python.sh $SCRIPT_DIR/../tests/python/jax_auto_task_test.py  --gpus 2
$SCRIPT_DIR/run-python.sh $SCRIPT_DIR/../tests/python/jax_auto_task_test.py  --cpus 4 --gpus 0
$SCRIPT_DIR/run-python.sh $SCRIPT_DIR/../tests/python/jax_auto_task_test.py  --cpus 8 --gpus 0

