import argparse
import gc
import os
import sys
from typing import Optional, Sequence


def init(
    disable_gc: Optional[bool] = None,
    cpus: Optional[int] = None,
    gpus: Optional[int] = None,
    fbmem: int = 8000,
    sysmem: int = 4000,
    zcmem: int = 32,
    debug: Optional[str] = None,
    network: str = "none",
    kthreads: bool = False,
    profile: bool = False,
    distributed: bool = False,
    dump: Optional[str] = None,
    dump_all_passes: bool = False,
    realm_argv: Optional[list[str]] = None,
    coordinator_address: str | None = None,
    num_processes: int = 1,
    process_id: int = 0,
    local_device_ids: int | Sequence[int] | None = None,
    cluster_detection_method: str | None = None,
    initialization_timeout: int = 300,
    coordinator_bind_address: str | None = None,
) -> None:
    # if cpus is explicitly specified and gpus is not
    # assume there are no GPUs on the system
    if cpus is not None and gpus is None:
        gpus = 0

    new_xla_flags = []
    if dump is not None:
        new_xla_flags.append(f"--xla_dump_to={dump}")
        if dump_all_passes:
            new_xla_flags.append("--xla_dump_hlo_pass_re=.*")
    existing_xla_flags = os.environ.get("XLA_FLAGS") or ""
    if new_xla_flags:
        os.environ["XLA_FLAGS"] = (
            " ".join(new_xla_flags) + " " + existing_xla_flags
        )
    if realm_argv is None:
        realm_argv = []
    if debug is not None:
        realm_debug_levels = {
            "info": 2,
            "debug": 1,
            "spew": 0,
        }
        level = realm_debug_levels[debug]
        realm_argv.append("-level")
        realm_argv.append(f"legate.xla={level}")

        xla_debug_levels = {
            "info": 1,
            "debug": 3,
            "spew": 5,
        }
        existing_vmodule = os.environ.get("TF_CPP_VMODULE") or ""
        vmods = [
            "legate_ifrt_client",
            "hlo_partition",
            "legate_pjrt_executable",
            "legate_pjrt_buffer",
            "legate_pjrt_client",
            "legate_store_cache",
            "gpu_executable",
            "loop_scheduler",
        ]
        xla_debug = xla_debug_levels[debug]
        vmods = [f"{vmod}={xla_debug}" for vmod in vmods]
        os.environ["TF_CPP_VMODULE"] = ",".join(vmods) + "," + existing_vmodule
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if cpus is not None:
        xla_flags += f"  --xla_force_host_platform_device_count={cpus}"
    os.environ["XLA_FLAGS"] = xla_flags

    existing_platforms = os.environ.get("JAX_PLATFORMS") or ""
    all_platforms = ["legate"]
    if existing_platforms:
        all_platforms.append(existing_platforms)
    all_platforms.append("cpu")
    os.environ["JAX_PLATFORMS"] = ",".join(all_platforms)

    import legate.jax

    legate.jax.set_startup_config(
        cpus=cpus,
        gpus=gpus,
        fbmem=fbmem,
        zcmem=zcmem,
        sysmem=sysmem,
        network=network,
        kthreads=kthreads,
        argv=realm_argv,
        profile=profile,
    )

    if disable_gc:
        gc.enable()

    import jax

    import legate.jax

    jax.lax.with_sharding_constraint = legate.jax.with_sharding_constraint

    if distributed:
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
            local_device_ids=local_device_ids,
            cluster_detection_method=cluster_detection_method,
            initialization_timeout=initialization_timeout,
            coordinator_bind_address=coordinator_bind_address,
        )


def init_test():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="the number of cpus to run the test on",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="the number of cpus to run the test on",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="none",
        help="the network layer to use for the tests",
    )
    parser.add_argument(
        "--debug",
        type=str,
        default=None,
        choices=["info", "debug", "spew"],
        help="the debug level",
    )
    parser.add_argument(
        "--kthreads",
        action="store_true",
        default=False,
        help="whether to use kernel or user threads",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        default=False,
        help="whether to dump HLO modules from tests",
    )
    parser.add_argument(
        "--dump-all-passes",
        action="store_true",
        default=False,
        help="whether to dump all intermediate HLO modules from tests",
    )

    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if args.dump:
        dump = "dump"
    else:
        dump = None

    init(
        cpus=args.cpus,
        gpus=args.gpus,
        network=args.network,
        debug=args.debug,
        kthreads=args.kthreads,
        dump=dump,
        dump_all_passes=args.dump_all_passes,
    )
