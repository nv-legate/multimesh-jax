#! /usr/bin/env python

import argparse
import glob
import math
import os
import re
import runpy
import sys
from pathlib import Path
from typing import Optional

try:
    from jax_plugins.legate import init
except ImportError:
    pass



parser = argparse.ArgumentParser(allow_abbrev=False)


## LEGION parameters
################################################################
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
    "--zcmem",
    type=int,
    default=4,
    help="The amount in GB of zero-copy (host pinned) memory to use",
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
    "--no-physical-tracing",
    action="store_true",
    default=False,
    help="Whether to only perform logical tracing, not full physical tracing",
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




## XLA parameters
################################################################
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
xla.add_argument(
    "--no-latency-hiding-scheduler", dest="latency_hiding_scheduler", action="store_false"
)
xla.add_argument(
    "--no-xla-pipelining", dest="xla_pipelining", action="store_false"
)
xla.add_argument(
    "--no-xla-loop-buffering", dest="xla_loop_buffering", action="store_false"
)
xla.add_argument(
    "--xla-ar-threshold", type=int, default=4294967296,
)
xla.add_argument(
    "--xla-ag-threshold", type=int, default=4294967296,
)
xla.add_argument(
    "--xla-rs-threshold", type=int, default=33554432,
)



## XLA parameters
################################################################
legate_jax = parser.add_argument_group("Legate-Jax")
legate_jax.add_argument(
    "--backend",
    type=str,
    choices=["cuda", "legate", "cpu"],
    help="The JAX backend to use",
    default="legate",
)
legate_jax.add_argument(
    "--autoshard",
    action=argparse.BooleanOptionalAction,
    default=True,
)
legate_jax.add_argument(
    "--debug",
    type=str,
    default=None,
    choices=["info", "debug", "spew"],
    help="The debug level",
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
legate_jax.add_argument(
    "--dump",
    type=str,
    default=None,
    help="A folder for dumping the HLO modules",
)
legate_jax.add_argument(
    "--dump-all-passes",
    action="store_true",
    default=False,
    help="Whether to dump all intermediate HLO modules",
)

legate_jax.add_argument(
    "--replicate-small-params",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Replicate all smaller than batch*squence_length",
)


maxtext = parser.add_argument_group("MaxText")

maxtext.add_argument(
    "--maxtext-config",
    type=str,
    default=None,
    help="The path to the yaml config",
)

maxtext.add_argument(
    "--model-name",
    type=str,
    default=None,
    help="Override model name in case base config is selected",
)

maxtext.add_argument(
    "--num-steps",
    type=int,
    default=None,
    help="The number of training steps to run",
)

maxtext.add_argument(
    "--base-num-decoder-layers",
    type=int,
    default=None,
    help="The number of transformer layers",
)

maxtext.add_argument(
    "--per-device-batch-size",
    type=int,
    default=2,
    help="The batchsize per device",
)

maxtext.add_argument(
    "--max-target-length",
    type=int,
    default=2048,
    help="The maximum sequence length",
)

maxtext.add_argument(
    "--attention",
    type=str,
    default=None,
    choices=['autoselected', 'dot_product', 'flash', 'cudnn_flash_te'],
    help="Override attention with one of valid attention types",
)

maxtext.add_argument(
    "--attention-type",
    type=str,
    default=None,
    choices=['global', 'local_sliding'],
    help="Override attention-type  with one valid types",
)

maxtext.add_argument(
    "--remat-policy",
    type=str,
    default=None,
    choices=['minimal', 'save_dot_except_mlpwi', 'save_dot_except_mlp', 'save_qkv_proj', 'qkv_proj_offloaded', 'minimal_offloaded', 'save_out_proj', 'full'],
    help="Override remat_policy",
)

maxtext.add_argument(
    "--compile-topology-num-slices",
    type=int,
    default=None,
    help="Override slice topology",
)

maxtext.add_argument(
    "--use-iota-embed",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Trigger use_iota_embed, default is 'false'",
)

maxtext.add_argument(
    "--logits-dot-in-fp32",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Trigger logits_dot_in_fp32, default is 'false'",
)

maxtext.add_argument(
    "--scan-layers",
    dest="scan_layers",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Whether to wrap repeated layers in a while-loop (True) or unroll (False)",
)

#####################################################################
args = parser.parse_args()

if args.gpus > 0:
    if args.nodes > 1:
        hardware = 'gpu_multiprocess'
    else:
        hardware = 'gpu'
else:
    hardware = 'cpu'

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
    if args.batch_size is None:
        raise ValueError(
            "must give explicit --batch-size when using --dump-only"
        )


## setup environment variables
#####################################################################
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
    "hlo_memory_scheduler",
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
    f"--xla_gpu_enable_nccl_comm_splitting={str(args.use_nccl_comm_split).lower()}",  # noqa: E501
    f"--xla_force_host_platform_device_count={args.cpus}",
    f"--xla_gpu_enable_latency_hiding_scheduler={args.latency_hiding_scheduler}",
    "--xla_gpu_enable_triton_gemm=false",
    "--xla_gpu_graph_level=0",
]


# these come from the nvidia JAX toolbox benchmarks
if args.backend == "legate":
    xla_flags.extend(
        [
            f"--xla_gpu_all_reduce_combine_threshold_bytes={args.xla_ar_threshold}",
            f"--xla_gpu_all_gather_combine_threshold_bytes={args.xla_ag_threshold}",
            f"--xla_gpu_reduce_scatter_combine_threshold_bytes={args.xla_rs_threshold}",
            f"--xla_gpu_enable_pipelined_all_gather={args.xla_pipelining}",
            f"--xla_gpu_enable_pipelined_reduce_scatter={args.xla_pipelining}",
            f"--xla_gpu_enable_pipelined_all_reduce={args.xla_pipelining}",
            f"--xla_gpu_enable_while_loop_double_buffering={args.xla_loop_buffering}",
            "--xla_gpu_enable_all_gather_combine_by_dim=false",
            "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
            # these are specific to legate
            "--xla_gpu_enable_highest_priority_async_stream=true",
            "--xla_gpu_enable_triton_softmax_fusion=false",
            #"--xla_gpu_all_reduce_combine_threshold_bytes=51200",
            #"--xla_dump_hlo_pass_re=.*",
        ]
    )
else:
    xla_flags.extend(
        [
            f"--xla_gpu_all_reduce_combine_threshold_bytes={args.xla_ar_threshold}",
            f"--xla_gpu_all_gather_combine_threshold_bytes={args.xla_ag_threshold}",
            f"--xla_gpu_reduce_scatter_combine_threshold_bytes={args.xla_rs_threshold}",
            f"--xla_gpu_enable_pipelined_all_gather={args.xla_pipelining}",
            f"--xla_gpu_enable_pipelined_reduce_scatter={args.xla_pipelining}",
            f"--xla_gpu_enable_pipelined_all_reduce={args.xla_pipelining}",
            f"--xla_gpu_enable_while_loop_double_buffering={args.xla_loop_buffering}",
            "--xla_gpu_enable_all_gather_combine_by_dim=false",
            "--xla_gpu_enable_reduce_scatter_combine_by_dim=false",
            ## added
            "--xla_gpu_enable_triton_softmax_fusion=false",
        ]
    )


if args.collective_matmul is not None:
    xla_flags.extend(
        [
            f"--xla_gpu_threshold_for_windowed_einsum_mib={args.collective_matmul}",  # noqa: E501
            "--xla_gpu_multi_streamed_windowed_einsum=true",
            "--xla_gpu_use_memcpy_local_p2p=true",
        ]
    )


if args.dump_only and args.dump is None:
    raise ValueError(
        "--dump-only requsted, but not HLO dump folder passed to --dump"
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

# valiate parallelism / device count
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

# 2 perdevice given by smoke config
batch_size = total_devices * args.per_device_batch_size 
mb_size = args.microbatch_size or batch_size
per_device_batch_size = batch_size // total_devices
devices_per_stage = total_devices // args.pp
transformer_num_devices = devices_per_stage
num_stages_per_interleave = args.pp
num_stages = num_stages_per_interleave * args.interleave
# 8 layers given by smoke test
base_num_decoder_layers = args.base_num_decoder_layers
layers_per_stage = base_num_decoder_layers // num_stages
layers_per_interleave = base_num_decoder_layers // args.interleave


#
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


if args.pp > 1:
    print(f"Pipeline parallelism with pp = {args.pp} enabled")

if args.model_name is not None and any(args.base_num_decoder_layers, args.remat, args.attention, args.max_target_length):
  raise Exception("cannot specify both --model-name and override model parameters")

print(f"config = {args.maxtext_config}")
print(f"steps= {args.num_steps}")
print(f"per_device_batch_size = {args.per_device_batch_size}")
print(f"base_num_decoder_layers = {args.base_num_decoder_layers}")
print(f"max_target_length = {args.max_target_length}")
print(f"total_devices = {total_devices}")
print(f"devices_per_stage = {devices_per_stage}")
print(f"transformer_num_devices = {transformer_num_devices}")
print(f"num_stages_per_interleave = {num_stages_per_interleave}")
print(f"num_stages = {num_stages}")
print(f"logits_num_devices = {logits_num_devices}")
print(f"embeddings_num_devices = {embeddings_num_devices}")
print(f"batch_size = {batch_size}")
print(f"mb_size = {mb_size}")



for key, val in env.items():
    os.environ[key] = str(val)
    print(f'env {key}={val}')

sys.path.append("/opt/maxtext/MaxText/")

import legate.jax  # noqa: E402

# initialize legate.jax
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
    )
  

    devices = list(range(total_devices))
    embeddings_device_axes = ["x", "y", "z"] 

    transformer_axes = [
        ("data", "x"),
        ("stage", "y"),
        ("fsdp", "y"),
        ("fsdp_transpose", "y"),
        ("sequence", "z"),
        ("tensor", "z"),
        ("autoregressive", "z"),
    ]


    transformer_x_dim = args.dp
    transformer_y_dim = args.fsdp
    transformer_z_dim = args.tp
    transformer_mesh = [
        transformer_x_dim,
        transformer_y_dim,
        transformer_z_dim,
    ]


    embeddings_axes = transformer_axes
    embeddings_mesh = transformer_mesh
    # the embeddings have a few extra things, make sure arrays
    # are fully shared over the replica/data dimension
    embeddings_axes.append(("data", "z"))



    # register tasks
    from legate.jax import register_task
    import train

    if args.pp == 1:
        # just default transformer parallelism
        register_task(
            "default",
            devices=devices[: transformer_num_devices],
            dims=transformer_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=transformer_axes,
        )
    else: # pp > 1

        train.set_mb_config(mb_size, args.schedule, num_stages, args.interleave)
        layer_regex = re.compile(r"layers_(\d+)")
        def compute_devices(name: str):
            layer = int(layer_regex.search(name).groups()[0])
            if layers_per_interleave is not None:
                # layer offset within an interleave
                layer = layer % layers_per_interleave
            stage = layer // layers_per_stage
            offset = transformer_num_devices * stage
            stop = offset + transformer_num_devices
            print(f'MFR: name = {name}, layer = {layer},  offset = {offset}, stop = {stop}')
            sys.stdout.flush()
            sys.stderr.flush()

            return list(range(offset, stop))

        if False:
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
            "(token_embedder).*",
            devices=devices[: embeddings_num_devices],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
        )
        register_task(
            "(emb_lookup).*",
            devices=devices[: embeddings_num_devices],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
        )
        register_task(
            "(position_emb).*",
            devices=devices[: embeddings_num_devices],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
        )

        register_task(
            "(decoder_norm).*",
            devices=devices[-logits_num_devices :],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
        )
        register_task(
            "(final_ln).*",
            devices=devices[-logits_num_devices :],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
        )

        register_task(
            "(compute_loss).*",
            devices=devices[-logits_num_devices :],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
        )

        register_task(
            "(logits_dense).*",
            devices=devices[-logits_num_devices :],
            dims=embeddings_mesh,
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
        )
        
        #register_task(
        #    "default",
        #    devices=devices[: embeddings_num_devices],
        #    dims=embeddings_mesh,
        #    device_axes=embeddings_device_axes,
        #    logical_axes=embeddings_axes,
        #)
       
        register_task(
            "default",
            devices=devices,
            dims=[len(devices), 1, 1],
            device_axes=embeddings_device_axes,
            logical_axes=embeddings_axes,
        )

import maxtext_utils as mu
maxtext_base = Path(mu.__file__).parent

config = args.maxtext_config or maxtext_base / "configs" / "base.yml"

argv = [
    "this",
    str(config),
    "run_name=my_name",
    f"base_output_directory=${os.getcwd()}/logs",
    "dataset_type=synthetic",
    "enable_single_controller=False",
    "enable_checkpointing=False",
    f"scan_layers={args.scan_layers}",
    f"per_device_batch_size={args.per_device_batch_size}",
    f"base_num_decoder_layers={args.base_num_decoder_layers}",
    f"max_target_length={args.max_target_length}",
    f"use_iota_embed={args.use_iota_embed}",
    f"logits_dot_in_fp32={args.logits_dot_in_fp32}",
]

argv.append(f"hardware={hardware}")

argv.append(f"ici_data_parallelism={args.dp}")
argv.append(f"ici_tensor_parallelism={args.tp*args.pp}")
argv.append(f"ici_fsdp_parallelism={args.fsdp}")

# optional maxtext args/overrides
if args.model_name:
    argv.append(f"model_name={args.model_name}")
if args.num_steps:
    argv.append(f"steps={args.num_steps}")
if args.attention:
    argv.append(f"attention={args.attention}")
if args.attention_type:
    argv.append(f"attention_type={args.attention_type}")
if args.remat_policy:
    argv.append(f"remat_policy={args.remat_policy}")
if args.compile_topology_num_slices:
    argv.append(f"compile_topology_num_slices={args.compile_topology_num_slices}")


# always enable recomputation
with legate.jax.enable_recomputation(True):
    if args.replicate_small_params:
        legate.jax.replicate_parameters_smaller_than_num_elements(
            batch_size * args.max_target_length
        )

    if args.hlo is None:
        import jaxlib
        sys.argv = argv
        try:
            if args.autoshard:
                # invoke directly to preserve microbatch settings
                train.main(argv)
            else:
                runpy.run_path("/opt/maxtext/MaxText/train.py", run_name="__main__")
        except jaxlib.xla_extension.XlaRuntimeError as e:
            need_throw = True
            if args.dump_only:
                path = Path(args.dump)
                if path.exists():
                    globber = (
                        path
                        / "*pjit_autoshard_step*before_optimizations.hlo.pb"
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
            erase_sharding=args.erase_explicit_sharding,
            autoshard=args.autoshard,
            platform=platform,
            device_mem_gb=args.fbmem,
        )

