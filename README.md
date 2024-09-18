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


## Building a Legate-JAX container

Instructions for building Legate-JAX can be found in the [README](docker/README.md).
Included in the `docker` folder are scripts showing how to configure, build, and install
the various components.

### JAX and Jaxlib Compatibility


In the future, a standard JAX and Jaxlib installation should be compatible with Legate-JAX
if JAX/Jaxlib are the most recent version and XLA is top-of-tree for the plugin client.
For now, the Jaxlib will need to be installed from source for the custom XLA fork used to build the Legate-JAX client.

## Development workflows

Until more changes can be upstreamed, Legate-Jax will use a rebase model with the base branch updated regularly from the upstream Jax and XLA repositories.
