#! /usr/bin/env python

import argparse
import glob
import math
import os
import re
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gin

parser = argparse.ArgumentParser(allow_abbrev=False)

legion = parser.add_argument_group("Legion")

legion.add_argument(
    "--nodes",
    type=int,
    default=None,
    help="The number of nodes to run on",
)

legion.add_argument(
    "--network",
    type=str,
    choices=["none", "gasnetex", "ucx"],
    default="none",
    help="The Legion network module to use",
)

legion.add_argument(
    "--fbmem",
    type=int,
    default=70,
    help="The amount in GB of frame-buffer memory to use",
)

legion.add_argument(
    "--sysmem",
    type=int,
    default=4,
    help="The amount in GB of host memory to use",
)

legion.add_argument(
    "--eager-sysmem",
    type=int,
    default=None,
    help="The amount in GB of host memory to reserve for eager allocations",  # noqa: E501
)

legion.add_argument(
    "--eager-fbmem",
    type=int,
    default=None,
    help="The amount in GB of frame-buffer memory to reserve for eager allocations",  # noqa: E501
)

legion.add_argument(
    "--gpus",
    type=int,
    default=8,
    help="The number of GPUs to use per-node.",
)

legion.add_argument(
    "--cpus",
    type=int,
    default=4,
    help="The number of CPUs to use per-node.",
)

legion.add_argument(
    "--profile",
    type=str,
    default=None,
    help="The root of the profile file, if Legion profiling should be activated",  # noqa: E501
)

xla = parser.add_argument_group("XLA")

xla.add_argument(
    "--debug-nccl",
    action="store_true",
    default=False,
    help="Whether to print NCCL debug information",
)

xla.add_argument(
    "--use-nccl-comm-split",
    action="store_true",
    default=False,
    help="Whether to use comm split to create communicators",
)

xla.add_argument(
    "--collective-matmul",
    type=int,
    default=None,
    help="Specify the cutoff in MiB for activating collective matul/windowed einsum tensor parallelism",  # noqa: E501
)

legate_jax = parser.add_argument_group("Legate-Jax")

legate_jax.add_argument(
    "--backend",
    type=str,
    choices=["cuda", "legate", "cpu"],
    help="The JAX backend to use",
    default="legate",
)

legate_jax.add_argument(
    "--no-autoshard", dest="autoshard", action="store_false"
)
legate_jax.add_argument("--autoshard", dest="autoshard", action="store_true")


legate_jax.add_argument(
    "--debug",
    type=str,
    default=None,
    choices=["info", "debug", "spew"],
    help="The debug level",
)

legate_jax.add_argument(
    "--cache-parallelism",
    type=int,
    default=3,
    help="The min parallelism for the store cache. Higher levels improve perf, but increase mem usage",  # noqa: E501
)

legate_jax.add_argument(
    "--max-out-of-order",
    type=int,
    default=0,
    help="The maximum number of tasks that can run out-of-order at at time. 0 is unlimited.",  # noqa: E501
)

legate_jax.add_argument(
    "--load-balance-embeddings",
    action="store_true",
    default=False,
    help="Whether to rotate microbatches across different submeshes for load-balancing",  # noqa: E501
)

legate_jax.add_argument(
    "--erase-explicit-sharding",
    action="store_true",
    default=False,
    help="Whether to erase explicit sharding in the module and only use autosharding",  # noqa: E501
)

legate_jax.add_argument(
    "--max-replica-sharding",
    action="store_true",
    default=False,
    help="Whether to rearrange the task mesh to shard as much as possible over the replica dimension even when pure model parallelism is requested",  # noqa: E501
)

xla_debug_levels = {
    None: 0,
    "info": 1,
    "debug": 3,
    "spew": 5,
}

legate_jax.add_argument(
    "--pp",
    type=int,
    default=1,
    help="The degree of pipeline parallelism",
)

legate_jax.add_argument(
    "--tp",
    type=int,
    default=1,
    help="The degree of tensor parallelism",
)

legate_jax.add_argument(
    "--fsdp",
    type=int,
    default=1,
    help="The degree of fully-sharded data parallelism",
)

legate_jax.add_argument(
    "--microbatch-reshape",
    type=int,
    default=None,
    help="Reshape microbatches to take strided slices",
)

legate_jax.add_argument(
    "--enable-tracing",
    action="store_true",
    default=False,
    help="Whether to enable Legion tracing for the training step",
)

legate_jax.add_argument(
    "--split-large-traces",
    action="store_true",
    default=False,
    help="Whether to split large traces into smaller sub-traces",
)

legate_jax.add_argument(
    "--strict-static-order",
    action="store_true",
    default=False,
    help="Whether to force tasks to follow a pre-defined static order",
)

legate_jax.add_argument(
    "--dp",
    type=int,
    default=1,
    help="The degree of data parallelism",
)

legate_jax.add_argument(
    "--interleave",
    type=int,
    default=1,
    help="The amount of interleaving (circular scheduling)",
)

legate_jax.add_argument(
    "--distribute-embeddings",
    action="store_true",
    default=False,
    help="Whether to distribute embeddings computation across all GPUs or include in Layer 0",  # noqa: E501
)

legate_jax.add_argument(
    "--sequence-parallel",
    action="store_true",
    dest="sequence_parallel",
    help="Whether to use sequence parallelism",  # noqa: E501
)
legate_jax.add_argument(
    "--no-sequence-parallel",
    action="store_false",
    dest="sequence_parallel",
    help="Whether to use sequence parallelism",  # noqa: E501
)

legate_jax.add_argument(
    "--schedule",
    type=str,
    choices=["fill-drain", "gpipe", "1f1b", "wavefront"],
    default="fill-drain",
    help="The microbatch schedule to use",
)

legate_jax.add_argument(
    "--only-fuse-loop-tasks",
    action="store_true",
    default=False,
    help="Only fuse tasks inside loops",
)

legate_jax.add_argument(
    "--microbatch-size",
    type=int,
    default=None,
    help="The size of the microbatches to use. Default is to match the global batch size",  # noqa: E501
)

legate_jax.add_argument(
    "--hlo",
    type=str,
    default=None,
    help="Path to an HLO module to compile. This starts an HLO module compilation test rather than a full PaxML run",  # noqa: E501
)

legate_jax.add_argument(
    "--dump-hlo",
    action="store_true",
    default=False,
    help="Whether to dump HLO modules without full execution",
)

paxml = parser.add_argument_group("PaxML")

paxml.add_argument(
    "--paxml-config",
    type=str,
    default="paxml.contrib.gpu.scripts_gpu.configs.Lambada126M",
    help="The PaxML model spec to pass to --fdl_config",
)

paxml.add_argument(
    "--num-heads",
    type=int,
    default=12,
    help="The number of attention heads",
)

paxml.add_argument(
    "--sequence-length",
    type=int,
    default=2048,
    help="The sequence length",
)

paxml.add_argument(
    "--model-dims",
    type=int,
    default=3072,
    help="The size of the model (embedding) dimension",
)

paxml.add_argument(
    "--batch-size",
    type=int,
    default=None,
    help="The global batch size across all devices. Defaults to max(32, 4 * no. gpus)",  # noqa: E501
)

paxml.add_argument(
    "--no-fuse-embeddings",
    action="store_true",
    help="Whether to prevent the embeddings/logits layers from fusing with transformer layers",  # noqa: E501
)

paxml.add_argument(
    "--common-autosharding",
    action="store_true",
    default=False,
    help="Whether all layers should share a common autosharding scheme mapping logical->device axes",  # noqa: E501
)

paxml.add_argument(
    "--vocab-path",
    type=str,
    default="model/model/c4_en_301_5Mexp2_spm.model",
    help="The path to the sentence pice model",
)

paxml.add_argument(
    "--precision",
    type=str,
    choices=["bfloat16", "float32"],
    default="bfloat16",
    help="The global batch size across all devices. Defaults to max(32, 4 * no. gpus)",  # noqa: E501
)

paxml.add_argument(
    "--num-steps",
    type=int,
    default=10,
    help="The number of training steps to run",
)

paxml.add_argument(
    "--remat",
    type=str,
    default="save_transformer_layer_output",
    choices=[
        "save_transformer_layer_output",
        "save_dot_with_no_batch_dims",
        "save_qkv_out_proj",
        "save_dot_only",
    ],
)

paxml.add_argument(
    "--num-layers",
    type=int,
    default=8,
    help="The number of transformer layers",
)

paxml.add_argument(
    "--optimizer",
    type=str,
    default="adam",
    choices=["adam", "sgd", "adafactor"],
    help="The type of optimizer to use",
)


xla = parser.add_argument_group("XLA")

paxml.add_argument(
    "--dump",
    type=str,
    default=None,
    help="A folder for dumping the HLO modules",
)

paxml.add_argument(
    "--dump-all-passes",
    action="store_true",
    default=False,
    help="Whether to dump all intermediate HLO modules",
)

args = parser.parse_args()


vmodule = [
    "legate_pjrt_buffer",
    "hlo_partition",
    "legate_computation",
    "legate_pjrt_client",
    "legate_pjrt_executable",
    "mpmd_input_output_buffer_alias",
    "legate_store_cache",
    "loop_scheduler",
    "legate_ifrt_client",
]

if args.debug_nccl:
    vmodule.append("nccl_utils")
    vmodule.append("nccl_collective_thunk")
    vmodule.append("nccl_api")


if args.dump_hlo:
    # forces a debug mode on the run where the HLO module
    # is generated from a single CPU run
    args.cpus = args.gpus
    args.gpus = 0
    if args.batch_size is None:
        raise ValueError(
            "must give explicit --batch-size when using --dump-hlo"
        )

xla_debug = xla_debug_levels[args.debug]

vmodule_str = ",".join([f"{root}={xla_debug}" for root in vmodule])
if custom_vmodule := os.environ.get("TF_CPP_VMODULE", None):
    vmodule_str = vmodule_str + "," + custom_vmodule

LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", "")
env = dict(
    VOCAB_PATH=args.vocab_path,
    JAX_PLATFORMS=args.backend,
    TF_CPP_MIN_LOG_LEVEL=0,
    TF_CPP_MAX_LOG_LEVEL=xla_debug,
    TF_CPP_VMODULE=vmodule_str,
    JAX_TRACEBACK_FILTERING="off",
    JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=0,
    LD_LIBRARY_PATH=f"{LD_LIBRARY_PATH}:/usr/local/cuda/lib64",
)

if args.backend == "legate":
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

xla_flags = [
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_enable_highest_priority_async_stream=true",
    "--xla_gpu_enable_triton_softmax_fusion=false",
    "--xla_gpu_all_reduce_combine_threshold_bytes=51200",
    "--xla_gpu_graph_level=0",
    f"--xla_gpu_enable_nccl_comm_splitting={str(args.use_nccl_comm_split).lower()}",  # noqa: E501
    f"--xla_force_host_platform_device_count={args.cpus}",
]

if args.collective_matmul is not None:
    xla_flags.extend(
        [
            f"--xla_gpu_threshold_for_windowed_einsum_mib={args.collective_matmul}",  # noqa: E501
            "--xla_gpu_multi_streamed_windowed_einsum=true",
            "--xla_gpu_use_memcpy_local_p2p=true",
        ]
    )

if args.dump_hlo and args.dump is None:
    raise ValueError(
        "--dump-only requested, but no HLO dump folder passed to --dump"
    )

if args.dump:
    xla_flags = xla_flags + [
        f"--xla_dump_to={args.dump}",
        "--xla_dump_hlo_as_text",
        "--xla_dump_hlo_as_proto",
    ]
if args.dump_all_passes:
    xla_flags = xla_flags + [
        "--xla_dump_hlo_pass_re=.*",
    ]

if existing_xla_flags := os.environ.get("XLA_FLAGS", None):
    xla_flags.append(existing_xla_flags)

env["XLA_FLAGS"] = " ".join(xla_flags)

if args.gpus == 0:
    if args.eager_sysmem is None:
        eager_alloc_percentage = 50
    else:
        eager_alloc_percentage = math.ceil(
            args.eager_sysmem * 100 / args.sysmem
        )
else:
    if args.eager_fbmem is None:
        eager_alloc_percentage = 50
    else:
        eager_alloc_percentage = math.ceil(args.eager_fbmem * 100 / args.fbmem)

num_nodes = args.nodes or 1
total_parallelism = args.tp * args.pp * args.dp * args.fsdp
if args.gpus == 0:
    devices_per_node = args.cpus
else:
    devices_per_node = args.gpus
total_devices = devices_per_node * num_nodes
if total_parallelism != total_devices:
    raise ValueError(
        f"PP={args.pp} DP={args.dp} TP={args.tp} FSDP={args.fsdp} does not multiply to total no. of GPUS {total_devices}"  # noqa: E501
    )

batch_size = args.batch_size or total_devices * 4
mb_size = args.microbatch_size or batch_size
per_core_batch_size = batch_size // total_devices
devices_per_stage = total_devices // args.pp
transformer_num_devices = devices_per_stage
num_stages_per_interleave = args.pp
num_stages = num_stages_per_interleave * args.interleave
layers_per_stage = args.num_layers // num_stages
layers_per_interleave = args.num_layers // args.interleave

if args.distribute_embeddings or args.load_balance_embeddings:
    logits_num_devices = total_devices
    embeddings_num_devices = total_devices
else:
    logits_num_devices = transformer_num_devices
    embeddings_num_devices = transformer_num_devices

if batch_size % total_devices:
    raise ValueError(
        f"No support for partial batches, gpus={total_devices} "
        f"does not divide batch_size={batch_size}"
    )

# use a fixed ratio for other parameters
hidden_dims = args.model_dims * 4
dims_per_head = args.model_dims // args.num_heads

for key, val in env.items():
    os.environ[key] = str(val)


@gin.configurable
@dataclass
class LambadaConfig:
    num_devices: int = -1
    transformer_num_devices: int = -1
    logits_num_devices: int = -1
    embeddings_num_devices: int = -1
    layers_per_stage: int = -1
    layers_per_interleave: Optional[int] = None

    def __call__(self):
        from legate.jax import register_task, register_task_factory

        devices = list(range(self.num_devices))

        layer_regex = re.compile(r"layers_(\d+)")

        microbatch_size = args.microbatch_size or batch_size
        if microbatch_size < args.fsdp:
            raise Exception(
                "FSDP parallelism cannot exceed the microbatch size"
            )

        if args.load_balance_embeddings:
            loop_dependent_embeddings_submesh_size = transformer_num_devices
            loop_dependent_logits_submesh_size = transformer_num_devices
            embeddings_submesh_num_devices = transformer_num_devices
        else:
            loop_dependent_embeddings_submesh_size = None
            loop_dependent_logits_submesh_size = None
            embeddings_submesh_num_devices = self.embeddings_num_devices

        transformer_x_dim = args.dp
        transformer_y_dim = args.fsdp
        transformer_z_dim = args.tp
        if (
            transformer_x_dim * transformer_y_dim * transformer_z_dim
            != transformer_num_devices
        ):
            raise Exception(
                "DP * FSDP * TP does not match no. devices in pipeline stage: "
                f"{transformer_num_devices}"
            )

        transformer_axes = [
            ("replica", "x"),
            ("data", "y"),
            ("mdl", "z"),
            # we minimally need to shard on z dimension here
            # to ensure that input batches are fully sharded
            ("seq", "z"),
        ]
        transformer_mesh = [
            transformer_x_dim,
            transformer_y_dim,
            transformer_z_dim,
        ]

        # try to shard as much as possible over the batch dimension
        if (
            args.max_replica_sharding
            and transformer_x_dim < microbatch_size
            and transformer_y_dim == 1
        ):
            rescale = min(
                transformer_z_dim, microbatch_size // transformer_x_dim
            )
            transformer_x_dim *= rescale
            transformer_z_dim //= rescale
            transformer_axes.append(("mdl", "x"))

        max_embedding_x_dim = embeddings_submesh_num_devices // args.tp
        embedding_x_dim = min(
            microbatch_size,
            embeddings_submesh_num_devices,
            max_embedding_x_dim,
        )

        if args.sequence_parallel:
            embedding_y_dim = (
                embeddings_submesh_num_devices // embedding_x_dim // args.tp
            )
            embedding_z_dim = args.tp
            # favor the seq dimension when sharding activations
            # shard batch dimension on x-axis
            # shard sequence dimension on y-axis and z-axis
            # shard vocab dimension on z-axis
            embeddings_axes = [
                ("replica", "x"),
                ("seq", "y"),
                ("seq", "z"),
                ("mdl", "z"),
                ("replica", "y"),
                ("replica", "z"),
            ]
        else:
            embedding_y_dim = 1
            embedding_z_dim = embeddings_submesh_num_devices // embedding_x_dim
            # shard batch and hidden dimensions on x-axis
            # shard vocab dimension on z-axis
            embeddings_axes = [
                ("replica", "x"),
                ("data", "y"),
                ("mdl", "z"),
                # we minimally need to shard on z dimension here
                # to ensure that input batches are fully sharded
                ("seq", "z"),
            ]

        if args.common_autosharding:
            embeddings_axes = transformer_axes
            embeddings_mesh = [
                transformer_x_dim,
                transformer_y_dim,
                transformer_z_dim,
            ]
            embeddings_device_axes = ["x", "y", "z"]
            # the embeddings have a few extra things, make sure arrays
            # are fully shared over the replica/data dimension
            embeddings_axes.append(("data", "z"))
        else:
            embeddings_mesh = [
                embedding_x_dim,
                embedding_y_dim,
                embedding_z_dim,
            ]
            embeddings_device_axes = ["x", "y", "z"]

        def compute_devices(name: str):
            layer = int(layer_regex.search(name).groups()[0])
            if self.layers_per_interleave is not None:
                # layer offset within an interleave
                layer = layer % self.layers_per_interleave
            stage = layer // self.layers_per_stage
            offset = self.transformer_num_devices * stage
            stop = offset + self.transformer_num_devices
            return list(range(offset, stop))

        if args.no_fuse_embeddings:
            fusion_color = 42
        else:
            fusion_color = 0

        register_task_factory(
            r"(layers_\d+)",
            device_callback=compute_devices,
            dims=transformer_mesh,
            device_axes=["x", "y", "z"],
            logical_axes=transformer_axes,
            fusion_color=fusion_color,
        )

        register_task(
            "(emb_lookup).*",
            devices=devices[: self.embeddings_num_devices],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
            loop_submesh_size=loop_dependent_embeddings_submesh_size,
        )

        register_task(
            "(position_emb).*",
            devices=devices[: self.embeddings_num_devices],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
            loop_submesh_size=loop_dependent_embeddings_submesh_size,
        )

        register_task(
            "(final_ln).*",
            devices=devices[-self.logits_num_devices :],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
            loop_submesh_size=loop_dependent_logits_submesh_size,
            loop_submesh_reverse=True,
        )

        register_task(
            "(compute_loss).*",
            devices=devices[-self.logits_num_devices :],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
            loop_submesh_size=loop_dependent_logits_submesh_size,
            loop_submesh_reverse=True,
        )

        register_task(
            "default",
            devices=devices[:embeddings_submesh_num_devices],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
        )


# this needs to come later after the env has been fully set up
# other jax will try and fail spectacularly to load the other platforms
import legate.jax  # noqa: E402

gin_config = f"""
import paxml.trainer_lib

ClientConfig:
  auto_shard = {args.autoshard}
  disable_gc = True

PaxLegateConfig:
  configurable = @LambadaConfig
  enable_tracing = {args.enable_tracing}
  local_mesh = ({args.dp}, {args.fsdp}, {args.tp}, 1)

LambadaConfig:
  num_devices = {total_devices}
  transformer_num_devices = {transformer_num_devices}
  logits_num_devices = {logits_num_devices}
  embeddings_num_devices = {embeddings_num_devices}
  layers_per_stage = {layers_per_stage}
  layers_per_interleave = {layers_per_interleave}

MicrobatchConfig:
  size = {mb_size}
  batch_reshape = {args.microbatch_reshape}
  schedule = '{args.schedule}'
  num_stages = {num_stages}
  interleave = {args.interleave}
"""


gin.parse_config(gin_config)

if args.hlo:
    # When doing a compile-only test, the config
    # needs to be passed explicitly to the init function
    configurable = LambadaConfig
else:
    configurable = None

if args.backend == "legate":
    legate.jax.init(
        configurable=configurable,
        cpus=args.cpus,
        gpus=args.gpus,
        sysmem=args.sysmem * 1000,
        fbmem=args.fbmem * 1000,
        eager_alloc_percentage=eager_alloc_percentage,
        network=args.network,
        debug=args.debug,
        profile=args.profile,
    )

if args.dump_hlo:
    ici_mesh = "[1,1,1,1]"
    per_core_batch_size = batch_size
else:
    ici_mesh = f"[{args.dp},{args.fsdp},{args.tp * args.pp},1]"

argv = [
    "this",
    "--job_log_dir=logs",
    f"--fdl.NUM_LAYERS={args.num_layers}",
    f"--fdl.NUM_HEADS={args.num_heads}",
    f"--fdl.MODEL_DIMS={args.model_dims}",
    f"--fdl.HIDDEN_DIMS={hidden_dims}",
    f"--fdl.DIMS_PER_HEAD={dims_per_head}",
    f"--fdl_config={args.paxml_config}",
    f"--fdl.FPROP_DTYPE='{args.precision}'",
    "--fdl.LAMBADA_TRAIN=True",
    "--fdl.REMAT=True",
    f'--fdl.CHECKPOINT_POLICY="{args.remat}"',
    f"--fdl.SUMMARY_INTERVAL_STEPS={args.num_steps}",
    f"--fdl.MAX_STEPS={args.num_steps}",
    "--fdl.EVAL_INTERVAL_STEPS=0",
    f"--fdl.PERCORE_BATCH_SIZE={per_core_batch_size}",
    "--tfds_data_dir=datasets",
    "--mode=train",
    "--alsologtostderr",
]

if (not args.autoshard and not args.hlo) or args.backend != "legate":
    if args.pp > 1:
        raise ValueError(
            "cannot configure pipeline parallelism through native CUDA backend"
        )
else:
    argv.append("-enable_auto_sharding")

argv.append("--fdl.DCN_MESH_SHAPE=[1,1,1,1]")
if args.dump_hlo:
    argv.append(f"--fdl.ICI_MESH_SHAPE={ici_mesh}")
    per_core_batch_size = batch_size
else:
    argv.append(f"--fdl.ICI_MESH_SHAPE={ici_mesh}")

if num_nodes > 1 and not args.dump_hlo:
    argv.append("--multiprocess_gpu")

if args.optimizer == "adafactor":
    argv.append("--fdl.USE_ADAFACTOR=True")
elif args.optimizer == "sgd":
    argv.append("--fdl.USE_SGD=True")

# always enable recomputation
with legate.jax.enable_recomputation(
    True
) as A, legate.jax.only_fuse_loop_tasks(
    args.only_fuse_loop_tasks
) as B, legate.jax.split_large_traces(
    args.split_large_traces
) as C, legate.jax.store_cache_min_parallelism(
    args.cache_parallelism
) as D, legate.jax.max_out_of_order(
    args.max_out_of_order
) as E, legate.jax.strict_static_order(
    args.strict_static_order
):
    legate.jax.replicate_parameters_smaller_than_num_elements(
        batch_size * args.sequence_length
    )
    if args.hlo is None or args.dump_hlo:
        import jaxlib

        sys.argv = argv
        try:
            runpy.run_module("paxml.main", run_name="__main__")
        except jaxlib.xla_extension.XlaRuntimeError as e:
            need_throw = True
            if args.dump_hlo:
                path = Path(args.dump)
                if path.exists():
                    globber = (
                        path / "*pjit_autoshard*before_optimizations.hlo.pb"
                    )
                    matches = glob.glob(str(globber))
                    if matches:
                        print(
                            "Dump succeeded in generating "
                            f"{matches[0]}. Run done with error: {e}"
                        )
                        need_throw = False

            if need_throw:
                raise e
        except Exception as e:
            # if dumping the hlo, squash the exception
            if not args.dump_hlo:
                raise e
            else:
                print(e)

if args.hlo is not None:
    # restore the original GPU count
    if args.dump_hlo:
        args.gpus = args.cpus
    # sort of funky here, but we have to instantiate the client
    # to force custom call registration
    # the easiest way to instantiate is to print the device list
    import jax

    print(jax.devices())

    platform = "gpu" if args.gpus else "cpu"
    from paxml.partitioning import LegateMeshWrapper

    with LegateMeshWrapper.mode(LegateMeshWrapper.Mode.COMPILING):
        legate.jax.replicate_parameters_smaller_than_num_elements(
            batch_size * args.sequence_length
        )
        legate.jax.compile_hlo_module(
            args.hlo,
            num_partitions=total_devices,
            erase_sharding=args.erase_explicit_sharding,
            autoshard=args.autoshard,
            platform=platform,
            device_mem_gb=args.fbmem,
        )
