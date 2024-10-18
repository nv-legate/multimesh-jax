# PaxML

One of the main Jax frameworks is [PaxML](https://github.com/google/paxml),
which is packaged with a standard Jax installation in the
[Jax toolbox images](https://github.com/NVIDIA/JAX-Toolbox).
In the Legate-Jax images, PaxML has been modified from its upstream version
to support *MPMD autosharding*, which enables pipeline parallelism.
PaxML's use of a global mesh and partition specs has been replaced with
more general `Sharding` annotations on the tensors.
Auto-microbatching annotations have also been added to the training function.

## run.py

Setting up pipeline parallelism is straightforward, but can be tedious.
A [helper script](https://github.com/nv-legate/legate.jax/blob/main/examples/paxml/run.py) is
therefore included for pre-training transformer models that makes configuring combinations
of pipeline, tensor, data, and fully-sharded data parallelism
simpler.

### Configuring parallelism

The most important set of options for configuring parallelism are:

```
  --pp PP               The degree of pipeline parallelism
  --tp TP               The degree of tensor parallelism
  --fsdp FSDP           The degree of fully-sharded data parallelism
  --dp DP               The degree of data parallelism
  --interleave N        The amount of interleaving (circular scheduling)
  --microbatch-size MICROBATCH_SIZE
                        The size of the microbatches to use. Default is no microbatching.
```

### Configuring the model

PaxML requires setting a base config that specifies the model
and its parameters. The default config is
`paxml.tasks.lm.params.nvidia.NVIDIA1_3B`, which implements a
standard GPT-style decoder.

```
  --paxml-config PAXML_CONFIG
                        The PaxML model spec to pass to --fdl_config
  --num-heads NUM_HEADS
                        The number of attention heads
  --sequence-length SEQUENCE_LENGTH
                        The sequence length
  --model-dims MODEL_DIMS
                        The size of the model dimension
  --batch-size BATCH_SIZE
                        The global batch size across all devices. Defaults to max(32, 4 * no. gpus)
  --num-layers NUM_LAYERS
                        The number of transformer layers
  --num-steps NUM_STEPS
                        The number of training steps to run
  --precision {bfloat16,float32}
                        The training precision. Default is bfloat16
  --remat {save_transformer_layer_output,save_dot_with_no_batch_dims,save_qkv_out_proj,save_dot_only}
  --optimizer {adam,sgd,adafactor}
```

### Transformer engine
The Legate-Jax image includes [Transformer Engine](https://github.com/NVIDIA/TransformerEngine),
which be turned on by specifying:

```
  --te, --no-te         Whether to use TransformerEngine (default: False)
```

## Machine configuration

Legate-Jax relies on a runtime orchestration layer called Realm that
manages memory allocations, task scheduling, and data movement.
Realm pre-allocates memory and requires the no. of GPUS per node
and no. of nodes to be specified.

```
  --nodes NODES         The number of nodes to run on
  --fbmem FBMEM         The amount in GB of frame-buffer memory to use
  --zcmem ZCMEM         The amount in GB of zero-copy (host pinned) memory to use
  --sysmem SYSMEM       The amount in GB of host memory to use
  --gpus GPUS           The number of GPUs to use per-node.
  --cpus CPUS           The number of CPUs to use per-node.
```

### Known Issues

* Checkpointing: PaxML/orbax conflict on when loading checkpoints. While writing
checkpoints works, the PaxML version in the repo does not load the correct
metadata and crashes with an inscrutable error.

## Advanced options

### Load-balancing embeddings/logits

Legate-Jax can attempt to distribute the work associted with embeddings/logits,
which create bubbles in the pipeline if they are running exclusively on a single node.

```
  --load-balance-embeddings, --no-load-balance-embeddings
                        Whether to rotate microbatches across different submeshes for load-balancing (default: False)
```

### Sharding sequence vs. model axis

Beyond the batch dimension, activations can be
sharded along the sequence or model axis. Generally, sequence parallelism
should perform better, but this can be tuned.

```
  --sequence-parallel, --no-sequence-parallel
                        Whether to use sequence parallelism (default: True)
```

### Profiling

Legate-Jax creates NVTX markers showing the task pipeline and data resharding
between stages in the pipeline. Profiling has to enabled in the application
as well as running with nsys.

```
  --profile, --no-profile
                        whether nsys profiling events should be created
```

To run, NVTX must minimally be collected (cuda,cublas,etc can also be collected):

```bash
nsys profile -t nvtx /opt/legate-jax-examples/run.py --profile ...
```
