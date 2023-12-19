SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

$SCRIPT_DIR/run-python.sh $SCRIPT_DIR/../tests/python/jax_microbatch_test.py --gpus 1
$SCRIPT_DIR/run-python.sh $SCRIPT_DIR/../tests/python/jax_task_test.py --gpus 1
