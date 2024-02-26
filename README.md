# Legate-JAX

This repository contains the wrapper library used to connect XLA (JAX) and the Legate runtime.
The repository can be used in two different modes:

1. XLA plugin client: This creates an XLA plugin client that can be used in JAX. Legate-JAX
builds `liblegate_plugin.so`, which can then be imported by JAX.
2. HLO replay prototype: This builds a subset of XLA into a `liblegate_hlo_prototype.so`, which is imported by the Legate-XLA bridge in a HLO replay tool independent of JAX.

The HLO replay tool is intended only for performance optimization and development workflow.
Users are expected to use Legate-JAX through the plugin client.

## Building the XLA client

The Legate plugin client needs to include functionality from a customized XLA,
which is built using Bazel. To perform a standard CUDA installation, the user
can simply run:

```
$ python -m pip install .
```

in the source folder. The scikit-build (via CMake) will download all dependencies, drive the XLA Bazel build, and install the Python wheel to create the Legate-JAX plugin (`jax_plugins.legate`). The default pip installation currently needs to download from
repositories that require credentials. For development builds - or to avoid login credentials - the user can add extra options.

The following CMake flags are useful for the underlying scikit-build that
does the pip installation:

* `-DCPM_xla_SOURCE=<...>`: An already downloaded XLA source folder.
* `-DCPM_legate_core_SOURCE=<...>`: An already downloaded Legate C++ core.
* `-DLegion_USE_CUDA=(ON|OFF)`: Whether to build with CUDA support. The default is ON.

These options can be placed in an environment variable read by scikit-build:

```
$ SKBUILD_CONFIGURE_OPTIONS="-DCPM_xla_SOURCE=/src/xla ...." \
python -m pip install . -vv
```

The XLA build can take a very long time to finish so we recommend building with `-vv` so
that the progress is printed to the screen.

The [XLA fork](https://github.com/nv-legate/xla) has extra files
for building a Legate plugin client in `/xla/pjrt/legate/`.
The files there provide the necessary Legate-specific implementations of:

1. `PjRtClient`
1. `PjRtBuffer`
1. `PjRtLoadedExecutable`

### Development Builds

Users can either use an existing Legate installation or have the Legate-JAX build and install its own Legate. If using an existing Legate, users should specify the location using `Legate_ROOT`:

```
$ cmake -DLegate_ROOT=<...> -S . -B build
$ cmake --build build
```

Once the build is complete, an editable installation of the plugin client can be done:

```
$ SETUPTOOLS_ENABLE_FEATURES="legacy-editable" \
SKBUILD_CONFIGURE_OPTIONS="-Dlegate_core_ROOT=<...>" \
python -m pip install --editable . -vv
```

### JAX and Jaxlib Compatibility


In the future, a standard JAX and Jaxlib installation should be compatible with Legate-JAX,
if JAX/Jaxlib are the most recent version and XLA is top-of-tree for the plugin client.
For now, the Jaxlib will need to be installed from source for the custom XLA fork used to build the Legate-JAX client.
Download the modified [JAX source code](https://github.com/nv-legate/jax.git). In the JAX source folder, run

```
$ python build/build.py \
  --enable_cuda \
  --bazel_options=--override_repository=xla=<...>
```
After a long build, it will print a message like the following about
the generated wheel that can be installed:

```
To install the newly-built jaxlib wheel, run:
  pip install dist/jaxlib-0.4.14.whl --force-reinstall
```

## Running with the custom XLA client

Legate (and the underlying Legion layer) requires numerous settings to be configured
via the `LEGION_DEFAULT_ARGS` environment variable. For a single node with 2 GPUS, an example would be:

```
$ export LEGION_DEFAULT_ARGS="
 -lg:local 0 \
 -ll:cpu 4 \
 -ll:gpu 2 \
 -cuda:skipbusy \
 -ll:util 2 \
 -ll:csize 4000 \
 -ll:fsize 4000 \
 -ll:zsize 32 \
 -ll:networks none \
 -lg:eager_alloc_percentage 50"
 ```

 Consult the Legion documentation for full details on the Legion command-line options.
 Once the Legion arguments are configured, the user can run:

 ```
 $ JAX_PLATFORMS=legate python my_jax_program.py
 ```

## Development workflows

Until more changes can be upstreamed, Legate-Jax will use a rebase model with the base branch updated weekly from the upstream Jax and XLA repositories.
Development should still use pull requests for merging and reviews, though. To open a pull request to either the nv-legate XLA or JAX repos, do the following steps:

1. Rebase all changes onto the most recent `legate-main` branch. `legate-main` is updated weekly on Monday morning.
2. Open a pull request against `legate-main`. Only your additional commits should appear. If a large number of commits appears with a merge-base that is not the most recent `legate-main`, then the PR needs to be rebased.
If the pull request is not merged into `legate-main` by the next Monday, the PR will need to be rebased again onto the updated `legate-main`.

Despite the extra challenges introduced by weekly rebasing, this ensures the cleanest possible updating of the base branch and ensures Legate provides a linear history on top of the most recent XLA and Jax code.
The Legate XLA Bridge repo does not require rebasing since there is no upstream.
