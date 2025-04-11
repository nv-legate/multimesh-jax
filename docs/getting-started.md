# Getting Started

The simplest way to get started is by downloading one of the Legate-JAX containers.
There is a full development container which includes all source files and enables
rebuilding each component.

## Docker Images

The easiest way to get started is by building images
using [Legate-Jax workflows](https://github.com/nv-legate/legate-jax-workflows).

## Running Jupyter tutorials with Docker

The recommended way to run the examples is through Docker.
To launch a Jupyter notebook in the container for running on CPU 
that can be loaded in a local browser:

```bash
docker run \
  -w /opt/legate-jax/docs/notebooks \
  -p 8675:8675 \
  gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-jax-dev \
  jupyter notebook --allow-root --ip 0.0.0.0 --port=8675
```
The notebook will then be available at the link shown.  
If GPUs are available, then docker can be launched as:

```bash
docker run \
  -w /opt/legate-jax/docs/notebooks \
  -p 8675:8675 \
  --gpus <N> \ 
  gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-jax-dev \
  jupyter notebook --allow-root --ip 0.0.0.0 --port=8675
```
where `<N>` is the number of GPUs.

## Building Images

Instructions for building Legate-JAX can be found in the [README](https://github.com/nv-legate/legate.jax/blob/main/docker/README.md).
Included in the `docker` folder are scripts showing how to configure, build, and install
the various components.


