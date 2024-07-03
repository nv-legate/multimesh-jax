import gc
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Sequence,
    Type,
    TypeAlias,
    Union,
)

import gin
import jax
from jax._src.pjit import flatten_axis_resources
from jax.lax import with_sharding_constraint as lax_with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_flatten, tree_unflatten

from .legate_jax_impl import (
    clear_tasks,
    enable_fast_path as _enable_fast_path,
    enable_implicit_tasks as _enable_implicit_tasks,
    enable_only_fuse_loop_tasks as _enable_only_fuse_loop_tasks,
    enable_recomputation as _enable_recomputation,
    enable_task_fusion as _enable_task_fusion,
    enable_tracing as _enable_tracing,
    register_task,
    register_task_factory,
    split_large_traces as _split_large_traces,
    unregister_task,
)
from .no_op import no_op

_ignore_transforms = 0

_GIN_CONFIG_ENV = "LEGATE_GIN_CONFIG"

_auto_shard_enabled = False

_gc_disabled = False

ContextValue = Union[bool, int]


@contextmanager
def ignore_transforms(ignore: bool = True):
    global _ignore_transforms
    try:
        if ignore:
            if _ignore_transforms == 0:
                _enable_implicit_tasks(False)
            _ignore_transforms += 1
        yield
    finally:
        if ignore:
            _ignore_transforms -= 1
            if _ignore_transforms == 0:
                _enable_implicit_tasks(True)


@contextmanager
def _set_context_value(
    flag: ContextValue,
    enable_fxn: Callable[[ContextValue], None],
    context_value: list[ContextValue],
) -> None:
    current = flag
    try:
        enable_fxn(flag)
        context_value[0] = flag
        yield
    finally:
        enable_fxn(current)
        context_value[0] = current


@contextmanager
def enable_fast_path(enable: bool = True, context_value=[True]):
    with _set_context_value(enable, _enable_fast_path, context_value):
        yield


@contextmanager
def enable_task_fusion(enable: bool = True, context_value=[True]):
    with _set_context_value(enable, _enable_task_fusion, context_value):
        yield


@contextmanager
def enable_recomputation(enable: bool = True, context_value=[True]):
    with _set_context_value(enable, _enable_recomputation, context_value):
        yield


@contextmanager
def split_large_traces(split_large_traces: bool = True, context_value=[False]):
    with _set_context_value(
        split_large_traces, _split_large_traces, context_value
    ):
        yield


@contextmanager
def enable_tracing(enable: bool = True, context_value=[False]):
    with _set_context_value(enable, _enable_tracing, context_value):
        yield


@contextmanager
def only_fuse_loop_tasks(enable: bool = True, context_value=[False]):
    with _set_context_value(
        enable, _enable_only_fuse_loop_tasks, context_value
    ):
        yield


def should_ignore_transforms() -> bool:
    backend = jax._src.xla_bridge.get_backend()
    if _ignore_transforms > 0:
        return True

    backend = jax._src.xla_bridge.get_backend()
    return backend.platform != "legate"


def _canonicalize_axes(ndim: int, pspec: PartitionSpec):
    if isinstance(pspec, NamedSharding):
        pspec = pspec.spec
    axes = []
    # make sure there is an entry for every dimension in the tensor
    # and that each entry is a tuple
    for entry in pspec:
        if entry is None:
            axes.append(())
        elif isinstance(entry, str):
            axes.append((entry,))
        else:
            axes.append(entry)
    for _ in range(len(axes), ndim):
        axes.append(())
    return axes


def with_sharding_constraint(x: Any, axis_resources: Any):
    if _ignore_transforms:
        return lax_with_sharding_constraint(x, axis_resources)
    flat_args, arg_treedef = tree_flatten(x)
    axis_flat = flatten_axis_resources(
        "legate-jax", arg_treedef, axis_resources, tupled_args=True
    )
    if len(axis_flat) != len(flat_args):
        raise Exception(
            f"axis_resources of length {len(axis_flat)}"
            f" does not match args of length {len(flat_args)}"
        )

    def _mark_arg(pspec, arg):
        json_str = json.dumps(
            {
                "axes": _canonicalize_axes(len(arg.shape), pspec),
            }
        )
        mark_sharding = no_op(name="AutoSharding", config=json_str)
        return mark_sharding(arg)

    flat_marked_args = [
        _mark_arg(pspec, arg) for pspec, arg in zip(axis_flat, flat_args)
    ]
    return tree_unflatten(arg_treedef, flat_marked_args)


_DeviceIdList: TypeAlias = Sequence[int]
_DeviceAxisDims: TypeAlias = Sequence[int]
_DeviceAxisNames: TypeAlias = Sequence[str]
_DeviceFactory: TypeAlias = Callable[[str], Sequence[int]]
_LogicalAxisPairs: TypeAlias = Sequence[tuple[str, str]]
ImplicitTask: TypeAlias = tuple[
    str,
    _DeviceIdList | _DeviceFactory,
    _DeviceAxisDims,
    _DeviceAxisNames,
    _LogicalAxisPairs,
]


def optional_kwargs(**kwargs):
    subset_kwargs = {}
    for key, value in kwargs.items():
        if value is not None:
            subset_kwargs[key] = value
    return subset_kwargs


@gin.configurable
@dataclass
class ClientConfig:
    auto_shard: Optional[bool] = False
    disable_gc: Optional[bool] = True
    configurable: Optional[Type] = None
    tasks: List[ImplicitTask] = field(default_factory=list)


def _init_config(client_config: ClientConfig):
    global _gc_disabled
    global _auto_shard_enabled

    if client_config.auto_shard is client_config.auto_shard:
        _auto_shard_enabled = True
        jax.lax.with_sharding_constraint = with_sharding_constraint

    if client_config.disable_gc:
        _gc_disabled = True
        gc.disable()

    if client_config.configurable:
        task_configure = client_config.configurable()
        task_configure()

    for name, devices, dims, device_axes, logical_axes in client_config.tasks:
        if callable(devices):
            register_task_factory(
                name=name,
                device_factory=devices,
                dims=dims,
                device_axes=device_axes,
                logical_axes=logical_axes,
            )
        else:
            register_task(
                name,
                devices=devices,
                dims=dims,
                device_axes=device_axes,
                logical_axes=logical_axes,
            )


@contextmanager
def context(client_config: ClientConfig):
    global _gc_disabled
    global _auto_shard_enabled
    current_gc_disabled = _gc_disabled
    current_auto_shard_enabled = _auto_shard_enabled

    _init_config(client_config)

    yield

    if _gc_disabled and not current_gc_disabled:
        gc.enable()

    if _auto_shard_enabled and not current_auto_shard_enabled:
        jax.lax.with_sharding_constraint = lax_with_sharding_constraint

    for name, devices, dims, device_axes, logical_axes in client_config.tasks:
        unregister_task(name)

    if client_config.configurable:
        clear_tasks()


def init(
    config: os.PathLike | None = None,
    auto_shard: Optional[bool] = None,
    disable_gc: Optional[bool] = None,
    configurable: Optional[type] = None,
    cpus: int = 4,
    gpus: int = 0,
    fbmem: int = 8000,
    sysmem: int = 4000,
    eager_alloc_percentage: int = 50,
    debug: Optional[str] = None,
    network: str = "none",
    profile: Optional[str] = None,
) -> None:
    if config is None:
        config = os.environ.get(_GIN_CONFIG_ENV)
    if config is not None:
        gin.parse_config_file(str(config), print_includes_and_imports=True)

    # the indirection here is to avoid setting any parameters
    # that are unspecificed programmatically to make them overwritable by gin
    kwargs = optional_kwargs(
        auto_shard=auto_shard, disable_gc=disable_gc, configurable=configurable
    )

    cpus = cpus or 4
    gpus = gpus or 0

    legion_args = [
        "-lg:local",
        0,
        "-ll:cpu",
        cpus,
        "-ll:gpu",
        gpus,
        "-cuda:skipbusy",
        "-ll:util",
        2,
        "-ll:csize",
        sysmem,
        "-ll:fsize",
        fbmem,
        "-ll:zsize",
        32,
        "-ll:networks",
        network,
        "-lg:eager_alloc_percentage",
        eager_alloc_percentage,
    ]
    if debug is not None:
        debug_levels = {
            "info": 2,
            "debug": 1,
            "spew": 0,
        }
        level = debug_levels[debug]

        # lower means more output from legate
        # if any debug is active, set to active
        legion_args.append(f"-level legate.xla={level}")

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
    xla_flags += f"  --xla_force_host_platform_device_count={cpus}"
    os.environ["XLA_FLAGS"] = xla_flags

    client_config = ClientConfig(**kwargs)
    _init_config(client_config)
