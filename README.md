# MultiMesh for JAX

MultiMesh for JAX provides a framework for creating `task` contexts within jitted computations,
allowing different subcomputations to be placed on different GPU submeshes. These
`task` computations can be combined inside a global `jit` with data resharding across submeshes
occurring automatically. MultiMesh therefore enables pipeline parallelism to be easily expressed.
Standard Jax SPMD sharding idioms can be used within each `task`,
enabling full N-dimensional parallelism.
This repository contains a PjRt plugin and Python helper APIs.

## Getting Started

The easiest way to get started is by building containers
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
  -w /opt/workspace/multimesh-jax/docs/notebooks \
  -p 8675:8675 \
  <image> \
  jupyter notebook --allow-root --ip 0.0.0.0 --port=8675
```
The notebook will then be available at the link shown.  
If GPUs are available, then docker can be launched as:

```bash
docker run \
  -w /opt/workspace/multimesh-jax/docs/notebooks \
  -p 8675:8675 \
  --gpus <N> \ 
  <image> \
  jupyter notebook --allow-root --ip 0.0.0.0 --port=8675
```
where `<N>` is the number of GPUs.

## Running transformers in MaxText

The main framework integrated with MultiMesh for Jax is [MaxText](https://github.com/AI-Hypercomputer/maxtext.git).
Running and configuring MaxText can be challenging given the number of
options for specifying the models. To aid in running transformer models,
a helper script has been added with a basic set of options
for configuring parallelism in the [MultiMesh for Jax workflows](https://github.com/nv-legate/multimesh-jax-workflows/blob/release-v0.1/maxtext/run.py)
`run.py --help` will give the full set of options.

### Known Issues

* When running with MPMD sharding, Orbax checkpoints may not work correctly since some
processes will not have addressable shards.
* For running with external libraries like TransformerEngine, parallelism is only
valid when running with process/GPU rather than process/node.

## JAX and Jaxlib Compatibility

In the future, a standard JAX and Jaxlib installation should be compatible with MultiMesh.
For now, patches will have to be applied to Jax and Jaxlib to work with the MultiMesh for JAX client.

