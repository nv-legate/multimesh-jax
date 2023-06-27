# Building a custom jaxlib

## Prequisites

In order to build a custom jaxlib users need:

1. The `main` branch from this repository.
2. The `legate` branch from the (XLA fork)[https://gitlab-master.nvidia.com/legate/xla]. Note that the tips of these two branches should always be compatible.
3. The `legate-xla-stable` branch from (legate.core)[https://gitlab-master.nvidia.com/legate/legate.core.internal].
4. The (jax repository)[https://github.com/google/jax]. The latest commit hash to be confirmed compatible is `dd023e266e6616494cdd590959103ddc646109c4`.
5. A cudnn installation (8.5.0 confirmed) 

## Install legate.core conda environment

Generate the conda dependency file and create the environemnt. Make sure to disable conda compiler wrappers as they tend to cause issues in the jax build later.

```
$ scripts/generate-conda-envs.pybuild
$ conda env create -n legate -f scripts/environment-test-linux-py3.9-cuda11.8.yaml
$ source activate legate
$ export CXX="/usr/bin/g++"
$ export CC="/usr/bin/gcc"
$ export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Install legate.core 

Install legate.core with cuda support. Add/change flags as desired. Assign install location to `LEGATE_DIR` variable.

```
$ ./install.py --cuda --editable --debug --max-fields 2048
$ export LEGATE_DIR=<legate.core-src>/_skbuild/linux-x86_64-3.9/cmake-install
$ export LD_LIBRARY_PATH="$LEGATE_DIR/lib:$LD_LIBRARY_PATH"
$ export PATH=$LEGATE_DIR/bin:$PATH
```

## Install legate-xla-bridge

Install the legate-xla-bridge. Assign build/install location to `XLA_LEGATE_ROOT`. Also add to `LD_LIBRARY_PATH`. 

```
$ MAKE_PREFIX_PATH=$LEGATE_DIR cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
$ cmake --build build
$ export XLA_LEGATE_ROOT=<legate-xla-bridge-src>/build
$ export LD_LIBRARY_PATH="$XLA_LEGATE_ROOT/lib:$LD_LIBRARY_PATH"
```

## Build jaxlib

Prepare environment variables for bazel build

```
$ export XLA_REPO=<xla-clone>
$ export CUDNN_DIR=<cudnn-install>
$ export LD_LIBRARY_PATH="$CUDNN_DIR:$LD_LIBRARY_PATH"
$ export TEST_TMPDIR=<some directory for bazel cache>
$ export TMP=<some tmp directory>
```

Important: Modify `WORKSPACE` file to point to local xla repository.

Trigger the bazel-build
```
$ python build/build.py --enable_cuda --noenable_tpu --noenable_rocm --enable_plugin_device --target_cpu_features default --cuda_path /usr/local/cuda --cudnn_path $CUDNN_DIR
```

## Install jaxlib/JAX

```
$ pip install dist/jaxlib-0.4.9-cp39-cp39-manylinux2014_x86_64.whl --force-reinstall
$ pip install -e .
```

# Running JAX programs with legate

```
$ CUDA_VISIBLE_DEVICES=0 JAX_PLATFORM_NAME=plugin legate --gpus 1 simple.py
# enabling logging levels (xla && legate_xla)
$ TF_CPP_VMODULE='legate_pjrt_client=2,legate_pjrt_executable=2,legate_pjrt_buffer=2' TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES=0 JAX_PLATFORM_NAME=plugin legate --gpus 1 --logging legate.xla=1 simple.py
```

