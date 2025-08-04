# Getting Started

The simplest way to get started is by downloading one of the MultiMesh for JAX containers.
There is a full development container which includes all source files and enables
rebuilding each component.

## Docker Images

The easiest way to get started is by using the prebuilt container images:

```bash
docker pull ghcr.io/nv-legate/multimesh-jax:v0.2
```

Images can also be built using [MultiMesh for Jax workflows](https://github.com/nv-legate/multimesh-jax-workflows).

### Requirements

The prebuilt container is built from a CUDA 12.8
[base image](https://hub.docker.com/layers/nvidia/cuda/12.8.1-cudnn-devel-ubuntu22.04/images/sha256-61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719)
on Ubuntu 22.
For system compatibility, refer to the
[CUDA toolkit documentation](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions).


## Running Jupyter tutorials with Docker

The recommended way to run the examples is through Docker.
Note: prebuilt containers are not yet available for open-source.
To launch a Jupyter notebook in the container for running on CPU 
that can be loaded in a local browser:

```bash
docker run \
  -w /opt/workspace/multimesh-jax/docs/notebooks \
  -p 8675:8675 \
  ghcr.io/nv-legate/multimesh-jax:v0.2 \
  jupyter notebook --allow-root --ip 0.0.0.0 --port=8675
```
The notebook will then be available at the URL shown,
which is usually `http://127.0.0.1:8675/...` or `http://localhost:8675/...`.
If GPUs are available, then docker can be launched as:

```bash
docker run \
  -w /opt/workspace/multimesh-jax/docs/notebooks \
  -p 8675:8675 \
  --gpus <N> \ 
  ghcr.io/nv-legate/multimesh-jax:v0.2 \
  jupyter notebook --allow-root --ip 0.0.0.0 --port=8675
```
where `<N>` is the number of GPUs.

## Building Images

Instructions for building MultiMesh for JAX can be found in the [build monorepo](https://github.com/nv-legate/multimesh-jax-workflows)
that handles all of the components required for building the plugin.


