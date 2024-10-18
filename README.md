# Legate-JAX

This repository contains the wrapper library used to connect XLA (JAX) and the Legate runtime.
The repository can be used in two different modes:

## Getting Started

The easiest way to get started is by downloading either the development or release image of Legate-JAX

* [Development Images](https://gitlab-master.nvidia.com/legate/quickstart.internal/container_registry/69362)
* [Release Images](https://gitlab-master.nvidia.com/legate/quickstart.internal/container_registry/69363)

To get started, we recommend pulling the development container:

```bash
$ docker pull gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-jax-dev
```

## Docs

User documentation including API reference and Jupyter tutorials can be found
on the [Nvidia docs page](http://sw-mobile-docs/cllr/legate-jax/). 


## Building a Legate-JAX container

Instructions for building Legate-JAX can be found in the [README](docker/README.md).
Included in the `docker` folder are scripts showing how to configure, build, and install
the various components.

## Running Jupyter tutorials with Docker

The recommended way to run the examples is through Docker.
To launch a Jupyter notebook in the container for running on CPU 
that can be loaded in a local browswer:

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

## Running transformers in PaxML

The first framework integrated with Legate-Jax was PaxML.
Running and configuring PaxML can be challenging given the number of `fiddle`
options for specifying the models. To aid in running transformer models,
a helper script has been added with a basic set of options
for configuring parallelism. `run.py --help` will give the full set of options.
Consult the [PaxML user docs](http://sw-mobile-docs/cllr/legate-jax/paxml.html)
for an explanation of the main options.

### Known Issues

* Checkpointing: PaxML/orbax conflict on when loading checkpoints. While writing
checkpoints works, the PaxML version in the repo does not load the correct
metadata and crashes with an inscrutable error.

## JAX and Jaxlib Compatibility

In the future, a standard JAX and Jaxlib installation should be compatible with Legate-JAX
if JAX/Jaxlib are the most recent version and XLA is top-of-tree for the plugin client.
For now, the Jaxlib will need to be installed from source for the custom XLA fork used to build the Legate-JAX client.

