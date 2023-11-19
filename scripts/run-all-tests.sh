SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

$SCRIPT_DIR/run-py-tests.sh
$SCRIPT_DIR/run-cpp-tests.sh
$SCRIPT_DIR/run-jax-tests.sh --test pjit_test
$SCRIPT_DIR/run-jax-tests.sh --test lax_numpy_test
