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
    "--eager-fbmem",
    type=int,
    default=50,
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

legate_jax = parser.add_argument_group("Legate-Jax")

legate_jax.add_argument(
    "--debug",
    type=int,
    default=0,
    help="The debug level. Higher is more verbose output",
)

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
    "--dp",
    type=int,
    default=1,
    help="The degree of data parallelism",
)

legate_jax.add_argument(
    "--interleave",
    type=int,
    default=1,
    help="The debug level. Higher is more verbose output",
)

legate_jax.add_argument(
    "--distribute-embeddings",
    type=bool,
    default=False,
    help="Whether to distribute embeddings computation across all GPUs or include in Layer 0",  # noqa: E501
)

legate_jax.add_argument(
    "--schedule",
    type=str,
    choices=["fill-drain", "gpipe", "1f1b"],
    help="The microbatch schedule to use",
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
    "--dump-only",
    action="store_true",
    default=False,
    help="Whether to only dump HLO modules without full execution",
)

paxml = parser.add_argument_group("PaxML")

paxml.add_argument(
    "--num-heads",
    type=int,
    default=12,
    help="The number of attention heads",
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


vmodule = [
    "legate_pjrt_buffer",
    "hlo_partition",
    "legate_computation",
    "legate_pjrt_client",
    "legate_pjrt_executable",
    "loop_schedule",
    "legate_ifrt_client",
]

args = parser.parse_args()

if args.dump_only:
    # forces a debug mode on the run where the HLO module
    # is generated from a single CPU run
    args.cpus = 1
    args.gpus = 0
    args.pp = 1
    args.tp = 1
    args.dp = 1
    args.fsdp = 1
    args.nodes = 1

vmodule_str = ",".join([f"{root}={args.debug}" for root in vmodule])
LD_LIBRARY_PATH = os.environ.get("LD_LIBRARY_PATH", "")
env = dict(
    VOCAB_PATH=args.vocab_path,
    JAX_PLATFORMS="legate",
    TF_CPP_MIN_LOG_LEVEL=0,
    TF_CPP_MAX_LOG_LEVEL=args.debug,
    TF_CPP_VMODULE=vmodule_str,
    JAX_TRACEBACK_FILTERING="off",
    XLA_PYTHON_CLIENT_PREALLOCATE="false",
    JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=0,
    LD_LIBRARY_PATH=f"{LD_LIBRARY_PATH}:/usr/local/cuda/lib64",
)

xla_flags = [
    "--xla_gpu_enable_latency_hiding_scheduler=true",
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_simplify_all_fp_conversions",
    "--xla_gpu_enable_async_all_gather=true",
    "--xla_gpu_enable_async_reduce_scatter=true",
    "--xla_gpu_enable_highest_priority_async_stream=true",
    "--xla_gpu_enable_triton_softmax_fusion=false",
    "--xla_gpu_all_reduce_combine_threshold_bytes=51200",
    "--xla_gpu_graph_level=0",
    "--xla_gpu_enable_async_all_reduce=true",
    f"--xla_force_host_platform_device_count={args.cpus}",
]

if args.dump_only and args.dump is None:
    raise ValueError(
        "--dump-only requsted, but not HLO dump folder passed to --dump"
    )

if args.dump:
    xla_flags = xla_flags + [
        f"--xla_dump_to={args.dump}",
        "--xla_dump_hlo_as_text",
        "--xla_dump_hlo_as_proto",
        "--xla_dump_hlo_as_dot",
    ]


env["XLA_FLAGS"] = " ".join(xla_flags)


eager_alloc_percentage = math.ceil(args.eager_fbmem / args.fbmem)

legion_args = [
    "-lg:local",
    0,
    "-ll:py",
    0,
    "-ll:cpu",
    args.cpus,
    "-ll:gpu",
    args.gpus,
    "-ll:util",
    2,
    "-ll:csize",
    4000,
    "-ll:fsize",
    args.fbmem,
    "-ll:zsize",
    32,
    "-ll:networks",
    args.network,
    "-ll:ib_rsize",
    0,
    "-lg:eager_alloc_percentage",
    eager_alloc_percentage,
    "-cuda:skipbusy",
]


if args.profile:
    legion_args = legion_args + [
        "-lg:prof",
        1,
        "-lg:prof_logfile",
        f"{args.profile}_%s.gz",
    ]

if args.debug > 0:
    legion_args.append("-level")
    legion_args.append(f"-legate.xla={args.debug}")

env["LEGION_DEFAULT_ARGS"] = " ".join(map(str, legion_args))
print(env["LEGION_DEFAULT_ARGS"])

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

batch_size = args.microbatch_size or total_devices * 4
mb_size = args.microbatch_size or batch_size
per_core_batch_size = batch_size // total_devices
devices_per_stage = total_devices // args.pp
transformer_num_devices = devices_per_stage
num_stages_per_interleave = args.pp
num_stages = num_stages_per_interleave * args.interleave
layers_per_stage = args.num_layers // num_stages
layers_per_interleave = args.num_layers // args.interleave

if args.distribute_embeddings:
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

        def compute_devices(name: str):
            layer = int(layer_regex.search(name).groups()[0])
            if self.layers_per_interleave is not None:
                # layer offset within an interleave
                layer = layer % self.layers_per_interleave
            stage = layer // self.layers_per_stage
            offset = self.transformer_num_devices * stage
            stop = offset + self.transformer_num_devices
            return list(range(offset, stop))

        register_task_factory(
            r"(layers_\d+)",
            device_callback=compute_devices,
            dims=[self.transformer_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "x"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "(emb_lookup).*",
            devices=devices[: self.embeddings_num_devices],
            dims=[self.embeddings_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "x"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "(position_emb).*",
            devices=devices[: self.embeddings_num_devices],
            dims=[self.embeddings_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "x"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "(final_ln).*",
            devices=devices[-self.logits_num_devices :],
            dims=[self.logits_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "x"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "(compute_loss).*",
            devices=devices[-self.logits_num_devices :],
            dims=[self.logits_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "x"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )

        register_task(
            "default",
            devices=devices[: self.embeddings_num_devices],
            dims=[self.embeddings_num_devices, 1],
            device_axes=["x", "y"],
            logical_axes=[
                ("replica", "x"),
                ("mdl", "x"),
                ("data", "y"),
            ],
        )


import legate.jax  # noqa: E402

gin_config = f"""
import paxml.trainer_lib

ClientConfig:
  auto_shard = True
  disable_gc = True

AutoShardingClientConfig:
  configurable = @LambadaConfig

LambadaConfig:
  num_devices = {total_devices}
  transformer_num_devices = {transformer_num_devices}
  logits_num_devices = {logits_num_devices}
  embeddings_num_devices = {embeddings_num_devices}
  layers_per_stage = {layers_per_stage}
  layers_per_interleave = {layers_per_interleave}

MicrobatchConfig:
  size = {mb_size}
"""

gin.parse_config(gin_config)
if args.hlo:
    # When doing a compile-only test, the config
    # needs to be passed explicitly to the init function
    configurable = LambadaConfig
else:
    configurable = None
legate.jax.init(configurable=configurable)

argv = [
    "this",
    "--enable_auto_sharding",
    "--job_log_dir=logs",
    f"--fdl.NUM_LAYERS={args.num_layers}",
    f"--fdl.NUM_HEADS={args.num_heads}",
    f"--fdl.MODEL_DIMS={args.model_dims}",
    f"--fdl.HIDDEN_DIMS={hidden_dims}",
    f"--fdl.DIMS_PER_HEAD={dims_per_head}",
    "--fdl_config=paxml.contrib.gpu.scripts_gpu.configs.Lambada126M",
    f"--fdl.FPROP_DTYPE='{args.precision}'",
    "--fdl.LAMBADA_TRAIN=True",
    "--fdl.REMAT=True",
    '--fdl.CHECKPOINT_POLICY="save_transformer_layer_output"',
    f"--fdl.SUMMARY_INTERVAL_STEPS={args.num_steps}",
    f"--fdl.MAX_STEPS={args.num_steps}",
    "--fdl.EVAL_INTERVAL_STEPS=0",
    f"--fdl.ICI_MESH_SHAPE=[1,{total_devices},1]",
    "--fdl.DCN_MESH_SHAPE=[1,1,1]",
    f"--fdl.PERCORE_BATCH_SIZE={per_core_batch_size}",
    "--tfds_data_dir=datasets",
    "--mode=train",
    "--alsologtostderr",
]

if args.optimizer == "adafactor":
    argv.append("--fdl.USE_ADAFACTOR=True")
elif args.optimizer == "sgd":
    argv.append("--fdl.USE_SGD=True")

if args.hlo is None:
    import jaxlib

    sys.argv = argv
    try:
        runpy.run_module("paxml.main", run_name="__main__")
    except jaxlib.xla_extension.XlaRuntimeError as e:
        need_throw = True
        if args.dump_only:
            path = Path(args.dump)
            if path.exists():
                globber = (
                    path / "*pjit_autoshard_step*before_optimizations.txt"
                )
                matches = glob.glob(str(globber))
                if matches:
                    print(
                        "Dump seems to have succeeded in generating "
                        f"{matches[0]}. Run finished early with error: {e}"
                    )
                    need_throw = False

        if need_throw:
            raise e

else:
    platform = "gpu" if args.gpus else "cpu"
    legate.jax.compile_hlo_module(
        args.hlo,
        num_partitions=total_devices,
        erase_sharding=True,
        platform=platform,
        device_mem_gb=args.fbmem,
    )
