# Legate-XLA Bridge

This repository contains the wrapper library used to connect XLA (Jax) and the Legate runtime.
The repository can be used in two different modes:

1. XLA plugin client: This creates an XLA plugin client that can be used in Jax. The Legate-XLA bridge
needs to first build `liblegate_xla.so`, which can then be imported by XLA.
2. HLO replay prototype: This builds a subset of XLA into a `liblegate_hlo_prototype.so`, which is imported by the Legate-XLA bridge in a HLO replay tool independent of Jax.

The HLO replay tool is intended only for performance optimization and development workflow.
Users are expected to use Legate through the plugin client.

## Building the XLA client

The Legate plugin client needs to be incorporated into a custom jaxlib installation.
A few additional steps are required relative to a regular Jax installation:

### Download and Build the Legate-XLA bridge

Users can either use an existing Legate installation or have the Legate-XLA bridge build and install its own Legate. If using an existing Legate, users should specify the location using `Legate_ROOT`:

```
> cmake -DLegate_ROOT=<...> -S . -B build
> cmake --build build
> cmake --install build
```

### Download the Legate-compatible fork of XLA

Users should down the `legate` branch from the (XLA fork)[https://gitlab-master.nvidia.com/legate/xla].
There is no build step required here - only downloading the source.
The relevant files for building a Legate plugin client are all found in `/xla/pjrt/legate/`.
The files there provide the necessary Legate-specific implementations of:

1. `PjRtClient`
1. `PjRtBuffer`
1. `PjRtLoadedExecutable`

### Download and Build Jaxlib

After downloading Jax, the custom jaxlib should be built by 1) pointing to the custom XLA branch and 2) enabling the plugin client:

```
> XLA_LEGATE_ROOT=<...> \
python build/build.py \
  --enable_cuda \
  --enable_plugin_device \
  --bazel_options=--override_repository=xla=<...>
```

The command requires the root (either install or build tree) of Legate be provided in the environment variable `XLA_LEGATE_ROOT`. This allows XLA to import and depend on the Legate-XLA bridge built in the previous step.
The jaxlib default XLA repository must be overridden.
Once completed, the build will provide a Python wheel that can be installed that will have a Legate XLA client that can be selected by specifying `"plugin"` for the backend type.

