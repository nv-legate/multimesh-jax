import gc
import os
from typing import Optional, Sequence


def init(
    disable_gc: Optional[bool] = None,
    cpus: Optional[int] = None,
    gpus: Optional[int] = None,
    fbmem: int = 8000,
    sysmem: int = 4000,
    zcmem: int = 32,
    eager_alloc_percentage: int = 50,
    debug: Optional[str] = None,
    network: str = "none",
    profile: Optional[str] = None,
    distributed: bool = False,
    no_physical_tracing: bool = False,
    dump: Optional[str] = None,
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

    legion_args = [
        "-lg:local",
        0,
        "-cuda:skipbusy",
        "-ll:util",
        2,
        "-ll:csize",
        sysmem,
        "-ll:fsize",
        fbmem,
        "-ll:zsize",
        zcmem,
        "-ll:networks",
        network,
        "-lg:eager_alloc_percentage",
        eager_alloc_percentage,
    ]
    if cpus is not None:
        legion_args.append("-ll:cpu")
        legion_args.append(cpus)
    if gpus is not None:
        legion_args.append("-ll:gpu")
        legion_args.append(gpus)

    if no_physical_tracing:
        legion_args.append("-lg:no_physical_tracing")

    new_xla_flags = []
    if dump is not None:
        new_xla_flags.append(f"--xla_dump_to={dump}")
    existing_xla_flags = os.environ.get("XLA_FLAGS") or ""
    if new_xla_flags:
        os.environ["XLA_FLAGS"] = (
            " ".join(new_xla_flags) + " " + existing_xla_flags
        )

    if debug is not None:
        legion_debug_levels = {
            "info": 2,
            "debug": 1,
            "spew": 0,
        }
        level = legion_debug_levels[debug]

        # lower means more output from legate
        # if any debug is active, set to active
        legion_args.append(f"-level legate.xla={level}")

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
            "gpu_executable",
        ]
        xla_debug = xla_debug_levels[debug]
        vmods = [f"{vmod}={xla_debug}" for vmod in vmods]
        os.environ["TF_CPP_VMODULE"] = ",".join(vmods) + "," + existing_vmodule
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

    if profile is not None:
        legion_args.extend(
            [
                "-lg:prof",
                1,
                "-lg:prof_logfile",
                f"{profile}_%s.gz",
            ]
        )

    if network != "ucx":
        legion_args.extend(["-ll:ib_rsize", "0"])

    legion_args_str = (
        " ".join(map(str, legion_args))
        + " "
        + os.environ.get("LEGION_DEFAULT_ARGS", "")
    )
    os.environ["LEGION_DEFAULT_ARGS"] = legion_args_str

    xla_flags = os.environ.get("XLA_FLAGS", "")
    if cpus is not None:
        xla_flags += f"  --xla_force_host_platform_device_count={cpus}"
    os.environ["XLA_FLAGS"] = xla_flags

    existing_platforms = os.environ.get("JAX_PLATFORMS") or ""
    os.environ["JAX_PLATFORMS"] = (
        "legate," + existing_platforms if existing_platforms else "legate"
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
