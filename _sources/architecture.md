# MultiMesh for JAX Architeture Overview

MultiMesh works through a combination of compiler and runtime support
for multiple-program, multiple data (MPMD). In contrast to
single-program, multiple data (SPMD), MPMD
enables complex task pipelines with processor submeshes executing different
tasks within a workflow. In deep learning, SPMD strategies
encompass things like data parallelism, tensor parallelism,
or [fully-sharded data parallelism](https://arxiv.org/abs/2304.11277). Pipeline parallelism strategies
such as [1F1B](https://arxiv.org/abs/2104.04473) or [zero-bubble](https://arxiv.org/abs/2401.10241), however, require MPMD programs.
The MultiMesh stack augments Jax and XLA with the ability
to express MPMD pipelines. This MPMD parallelism can be combined
with existing supporting for SPMD parallelism to express strategies
like [3D parallelism](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/parallelisms.html). 
Here we show the basic architecture of MultiMesh
with [MaxText](https://github.com/AI-Hypercomputer/maxtext).


![MultiMesh Stack Diagram](./images/MultiMeshStackDiagram.png)

## Python APIs

MultiMesh relies on ahead-of-time compilation in Jax using `AUTO` shardings.
Rather than binding shards directly to a physical mesh prior to JIT compilation,
MultiMesh captures "logical shardings" for tensors and translates them
to submesh shardings within a global JIT computation. Jax model code
is either minimimally modified or not modified at all. At the framework-level
in MaxText, we apply modest patches for converting MaxText to use
the ahead-of-time, AUTO compilation. 

Generally, when `jax_plugins.multimesh.init` is invoked, it
patches Jax functions to map to MPMD-compatible versions 
(e.g. `with_sharding_contraints`). MultMesh-specific calls are then
inserted into the StableHLO module that gets passed to the compiler backend.

## PjRt Plugin

Most of the functionality of MultiMesh is contained within a 
[PjRt plugin](https://opensource.googleblog.com/2024/03/pjrt-plugin-to-accelerate-machine-learning.html).
Similar to the way CUDA and other device-specific backends are built and used,
MultiMesh is installed as a `jax_plugins` namespace module and loaded as the 
backend by jaxlib when `JAX_PLATFORMS=multimesh` is given.

### Compiler

Only the MultiMesh plugin can compile and run the modules produced
when MultiMesh is imported and initialized since non-standard
custom calls are introduced. The compiler splits the MPMD
HLO module into a pipeline of SPMD tasks with data movement
and scheduling across SPMD tasks handled by the underlying MPMD
runtime (see figure).

![MultiMesh Compiler Diagram](./images/MultiMeshCompilerPipeline.png)

### Sharded Array Runtime (Zuku)

Each task in the MPMD pipeline can execute across multiple processors
with internal SPMD parallelism. Instead of submitting single-processor
tasks and point-to-point communication to the runtime, the runtime
*gang-schedules* multi-processor operations and plans [*cross-mesh resharding*](https://arxiv.org/abs/2211.05322)
of sharded arrays from one submesh to another. Within the plugin,
this is handled by an in-house runtime called Zuku. The runtime
translates bulk, multi-processor operations into a computation and 
communication schedule for each processor.

### Realm

MultiMesh (and the runtime Zuku) do not directly interact with CUDA or
other device-specific libraries. Instead they manage operations
through a hardware-abstraction layer provided by the [Realm runtime](https://gitlab.com/StanfordLegion/legion) 
(distributd within the Legion project).
Realm provides control- and data-plane operations with a simple
event-based model that unifies all the different device libraries
into a common event loop.

## Patches for Jax and Jaxlib

At present, the MPMD shardings used are not natively supported by Jax and Jaxlib.
Patches are therefore currently required to support submeshes in Jax.
Efforts are underway to support this submesh sharding model within upstream Jax,
but more design and implementation work is required.

