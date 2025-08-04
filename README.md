# MultiMesh for JAX

MultiMesh for JAX provides a framework for creating `task` contexts within jitted computations,
allowing different subcomputations to be placed on different GPU submeshes. These
`task` computations can be combined inside a global `jit` with data resharding across submeshes
occurring automatically. MultiMesh therefore enables pipeline parallelism to be easily expressed.
Standard Jax SPMD sharding idioms can be used within each `task`,
enabling full N-dimensional parallelism.
This repository contains a PjRt plugin and Python helper APIs.

## Getting Started

The easiest way to get started is by using the prebuilt containers

```bash
$ docker pull ghcr.io/nv-legate/multimesh-jax:v0.2
```

Containers can be also be built
using [MultiMesh for Jax workflows](https://github.com/nv-legate/multimesh-jax-workflows).

## Docs

User documentation including [API reference](http://nv-legate.github.io/multimesh-jax/api.html)
, [Jupyter tutorials](http://nv-legate.github.io/multimesh-jax/tutorials.html),
and [architecture overview](http://nv-legate.github.io/multimesh-jax/architecture.html)
can be found on the [docs page](http://nv-legate.github.io/multimesh-jax).


## Running Jupyter tutorials with Docker

The recommended way to run the examples is through Docker.
To launch a Jupyter notebook in the container built using [the build workflows](https://github.com/nv-legate/multimesh-jax-workflows)
for running on CPU:

```bash
docker run \
  --mount type=bind,source=$(pwd)/docs/notebooks,target=/opt/notebooks \
  -w /opt/notebooks \
  -p 8675:8675 \
  ghcr.io/nv-legate/multimesh-jax:v0.2 \
  jupyter notebook --allow-root --ip 0.0.0.0 --port=8675
```
The notebook will then be available at the URL shown,
which is usually `http://127.0.0.1:8675/...` or `http://localhost:8675/...`.
If GPUs are available, then docker can be launched as:

```bash
docker run \
  --mount type=bind,source=$(pwd)/docs/notebooks,target=/opt/notebooks \
  -w /opt/notebooks \
  -p 8675:8675 \
  --gpus <N> \ 
  ghcr.io/nv-legate/multimesh-jax:v0.2 \
  jupyter notebook --allow-root --ip 0.0.0.0 --port=8675
```
where `<N>` is the number of GPUs.

### Requirements

The prebuilt container is built from a CUDA 12.8
[base image](https://hub.docker.com/layers/nvidia/cuda/12.8.1-cudnn-devel-ubuntu22.04/images/sha256-61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719)
on Ubuntu 22.
For system compatibility, refer to the
[CUDA toolkit documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions).

## Running transformers in MaxText

The main framework integrated with MultiMesh for Jax is [MaxText](https://github.com/AI-Hypercomputer/maxtext.git).
Running and configuring MaxText can be challenging given the number of
options for specifying the models. To aid in running transformer models,
a helper script has been added with a basic set of options
for configuring parallelism in the [MultiMesh for Jax workflows](https://github.com/nv-legate/multimesh-jax-workflows/blob/release-v0.2/docker/workspace/maxtext-scripts/run.py)
`run.py --help` will give the full set of options.

### Known Issues

* When running with MPMD sharding, Orbax checkpoints may not work correctly since some
processes will not have addressable shards.
* For running with external libraries like TransformerEngine, parallelism is only
valid when running with process/GPU rather than process/node.
* The plugin is experimental and may not work correctly if used outside of the tutorials or documented MaxText examples.


## JAX and Jaxlib Compatibility

In the future, a standard JAX and Jaxlib installation should be compatible with MultiMesh.
For now, patches will have to be applied to Jax and Jaxlib to work with the MultiMesh for JAX client.

