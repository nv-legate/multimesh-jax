#! /usr/bin/env python

import argparse
import glob
import os
import re
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from jax_plugins.legate import init
except ImportError:
    pass
import gin

parser = argparse.ArgumentParser(allow_abbrev=False)

realm = parser.add_argument_group("Realm")

realm.add_argument(
    "--nodes",
    type=int,
    default=None,
    help="The number of nodes to run on",
)

realm.add_argument(
    "--network",
    type=str,
    choices=["none", "gasnetex", "ucx"],
    default=None,
    help="The Legion network module to use",
)

realm.add_argument(
    "--fbmem",
    type=int,
    default=70,
    help="The amount in GB of frame-buffer memory to use",
)

realm.add_argument(
    "--zcmem",
    type=int,
    default=4,
    help="The amount in GB of zero-copy (host pinned) memory to use",
)

realm.add_argument(
    "--sysmem",
    type=int,
    default=4,
    help="The amount in GB of host memory to use",
)

realm.add_argument(
    "--gpus",
    type=int,
    default=8,
    help="The number of GPUs to use per-node.",
)

realm.add_argument(
    "--cpus",
    type=int,
    default=4,
    help="The number of CPUs to use per-node.",
)

realm.add_argument(
    "--profile",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="whether nsys profiling events should be created",  # noqa: E501
)

xla = parser.add_argument_group("XLA")

xla.add_argument(
    "--dump",
    type=str,
    default=None,
    help="A folder for dumping the HLO modules",
)

xla.add_argument(
    "--dump-all-passes",
    action="store_true",
    default=False,
    help="Whether to dump all intermediate HLO modules",
)

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
    "--autoshard", default=True, action=argparse.BooleanOptionalAction
)

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
    action=argparse.BooleanOptionalAction,
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
    "--dp",
    type=int,
    default=1,
    help="The degree of data parallelism",
)

legate_jax.add_argument(
    "--strict-static-order",
    action="store_true",
    default=False,
    help="Whether to force tasks to follow a pre-defined static order",
)

legate_jax.add_argument(
    "--host-offload-min-reuse-distance",
    type=int,
    default=0,
    help="The minimum reuse distance to trigger host-offload of intermediates. 0 indicates no offloading",  # noqa: E501
)

legate_jax.add_argument(
    "--interleave",
    type=int,
    default=1,
    help="The amount of interleaving (circular scheduling)",
)

legate_jax.add_argument(
    "--distribute-embeddings",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to distribute embeddings computation across all GPUs or include in Layer 0",  # noqa: E501
)

legate_jax.add_argument(
    "--sequence-parallel",
    action=argparse.BooleanOptionalAction,
    default=True,
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
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Only fuse tasks inside loops",
)

legate_jax.add_argument(
    "--disable-task-fusion",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Only fuse tasks inside loops",
)

legate_jax.add_argument(
    "--microbatch-size",
    type=int,
    default=None,
    help="The size of the microbatches to use. Default is no microbatching",  # noqa: E501
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
    default="paxml.tasks.lm.params.nvidia.NVIDIA1_3B",
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
    help="The training precision. Default is bfloat16",
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

paxml.add_argument(
    "--te",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Whether to use TransformerEngine",
)


args, realm_argv = parser.parse_known_args()

if args.te:
    if args.tp == 1:
        raise Exception(
            "TransformerEngine (--te) requires tensor parallelism (--tp) > 1"
        )
    os.environ["ENABLE_TE"] = "1"
    os.environ["ENABLE_TE_SP"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "1"


if args.host_offload_min_reuse_distance > 0 and args.gpus > args.cpus:
    raise Exception(
        f"host offloading requires at least as many CPUs as GPUs per process: {args.cpus} < {args.gpus}"  # noqa: E501
    )

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
    "--xla_disable_hlo_passes=rematerialization",
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


if args.dump_hlo:
    # forces a debug mode on the run where the HLO module
    # is generated from a single CPU run
    args.cpus = args.gpus
    args.gpus = 0
    if args.batch_size is None:
        raise ValueError(
            "must give explicit --batch-size when using --dump-hlo"
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
class PaxTransformerConfig:
    num_devices: int = -1
    transformer_num_devices: int = -1
    logits_num_devices: int = -1
    embeddings_num_devices: int = -1
    layers_per_stage: int = -1
    layers_per_interleave: Optional[int] = None

    def __call__(self):
        from legate.jax import register_task

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
        ]

        if args.sequence_parallel:
            transformer_axes.append(("seq", "z"))
            transformer_axes.append(("mdl", "z"))
        else:
            transformer_axes.append(("mdl", "z"))
            transformer_axes.append(("seq", "z"))

        transformer_mesh = [
            transformer_x_dim,
            transformer_y_dim,
            transformer_z_dim,
        ]

        embedding_x_dim = args.dp

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

        register_task(
            r"(layers_\d+)",
            callback=compute_devices,
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

PaxLegateConfig:
  configurable = @PaxTransformerConfig
  local_mesh = ({args.dp}, {args.fsdp}, {args.tp}, 1)

PaxTransformerConfig:
  num_devices = {total_devices}
  transformer_num_devices = {transformer_num_devices}
  logits_num_devices = {logits_num_devices}
  embeddings_num_devices = {embeddings_num_devices}
  layers_per_stage = {layers_per_stage}
  layers_per_interleave = {layers_per_interleave}

MicrobatchConfig:
  size = {mb_size}
  schedule = '{args.schedule}'
  num_stages = {num_stages}
  interleave = {args.interleave}
"""


gin.parse_config(gin_config)

if args.network is None:
    if num_nodes > 1:
        network = "ucx"
    else:
        network = "none"
else:
    network = args.network

if args.backend == "legate":
    init(
        cpus=args.cpus,
        gpus=args.gpus,
        sysmem=args.sysmem * 1000,
        fbmem=args.fbmem * 1000,
        zcmem=args.zcmem * 1000,
        network=args.network,
        debug=args.debug,
        profile=args.profile,
        realm_argv=realm_argv,
    )

if args.dump_hlo:
    ici_mesh = f"[{args.dp},1,1,1]"
    per_core_batch_size = batch_size // args.dp
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
    per_core_batch_size = batch_size // args.dp
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
) as B, legate.jax.store_cache_min_parallelism(
    args.cache_parallelism
) as C, legate.jax.strict_static_order(
    args.strict_static_order
) as D, legate.jax.host_offload_min_reuse_distance(
    args.host_offload_min_reuse_distance
) as E, legate.jax.enable_task_fusion(
    args.enable_task_fusion
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

    platform = "gpu" if args.gpus else "cpu"
    from paxml.partitioning import LegateMeshWrapper

    with LegateMeshWrapper.mode(LegateMeshWrapper.Mode.COMPILING):
        with legate.jax.tasks(configurable=PaxTransformerConfig()):
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
