import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Type, TypeAlias, Union

import gin
import jax
from jax._src.pjit import flatten_axis_resources
from jax.lax import with_sharding_constraint as lax_with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_flatten, tree_map, tree_unflatten

from .legate_jax_impl import (
    clear_tasks,
    enable_fast_path as _enable_fast_path,
    enable_implicit_tasks as _enable_implicit_tasks,
    enable_only_fuse_loop_tasks as _enable_only_fuse_loop_tasks,
    enable_recomputation as _enable_recomputation,
    enable_task_fusion as _enable_task_fusion,
    enable_tracing as _enable_tracing,
    set_host_offload_min_reuse_distance,
    set_max_out_of_order,
    set_store_cache_min_parallelism,
    set_strict_static_order,
    split_large_traces as _split_large_traces,
)
from .no_op import no_op

_ignore_transforms = 0

_auto_shard = False

ContextValue = Union[bool, int]


@contextmanager
def ignore_transforms(ignore: Optional[bool] = True):
    global _ignore_transforms
    if ignore is None:
        yield
    else:
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
    flag: Optional[ContextValue],
    enable_fxn: Callable[[ContextValue], None],
    context_value: list[ContextValue],
) -> None:
    if flag is None:
        yield
    else:
        current = context_value[0]
        try:
            enable_fxn(flag)
            context_value[0] = flag
            yield
        finally:
            enable_fxn(current)
            context_value[0] = current


@contextmanager
def enable_fast_path(enable: Optional[bool] = None, context_value=[True]):
    with _set_context_value(enable, _enable_fast_path, context_value):
        yield


@contextmanager
def enable_task_fusion(enable: Optional[bool] = None, context_value=[True]):
    with _set_context_value(enable, _enable_task_fusion, context_value):
        yield


@contextmanager
def enable_recomputation(enable: Optional[bool] = None, context_value=[True]):
    with _set_context_value(enable, _enable_recomputation, context_value):
        yield


@contextmanager
def split_large_traces(
    split_large_traces: Optional[bool] = None, context_value=[False]
):
    with _set_context_value(
        split_large_traces, _split_large_traces, context_value
    ):
        yield


@contextmanager
def enable_tracing(enable: Optional[bool] = None, context_value=[False]):
    with _set_context_value(enable, _enable_tracing, context_value):
        yield


@contextmanager
def only_fuse_loop_tasks(enable: Optional[bool] = None, context_value=[False]):
    with _set_context_value(
        enable, _enable_only_fuse_loop_tasks, context_value
    ):
        yield


@contextmanager
def store_cache_min_parallelism(
    parallelism: Optional[int] = None, context_value=[3]
):
    with _set_context_value(
        parallelism, set_store_cache_min_parallelism, context_value
    ):
        yield


@contextmanager
def max_out_of_order(
    max_out_of_order: Optional[int] = None, context_value=[2]
):
    with _set_context_value(
        max_out_of_order, set_max_out_of_order, context_value
    ):
        yield


@contextmanager
def strict_static_order(order: Optional[bool] = None, context_value=[True]):
    with _set_context_value(order, set_strict_static_order, context_value):
        yield


@contextmanager
def host_offload_min_reuse_distance(
    reuse_distance: Optional[int] = None, context_value=[0]
):
    if reuse_distance is None:
        yield
    else:
        with _set_context_value(
            reuse_distance, set_host_offload_min_reuse_distance, context_value
        ):
            yield


@contextmanager
def autoshard(autoshard: Optional[bool] = None):
    global _auto_shard
    if autoshard is not None:
        current = _auto_shard
        _auto_shard = autoshard
    yield
    if autoshard is not None:
        _auto_shard = current


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
    global _auto_shard

    if _ignore_transforms or not _auto_shard:
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
    configurable: Optional[Type] = None


@contextmanager
def tasks(configurable: Optional[Type] = None):
    clear_tasks()

    if configurable is not None:
        task_configure = configurable()
        task_configure()

    yield

    clear_tasks()


@contextmanager
def _context(
    _configurable=None,
    _autoshard: Optional[bool] = None,
    _strict_static_order: Optional[bool] = None,
    _enable_tracing: Optional[bool] = None,
    _host_offload_min_reuse_distance=None,
    _only_fuse_loop_tasks=None,
    _split_large_traces=None,
    _enable_task_fusion=None,
    _enable_fast_path=None,
    _ignore_transforms=None,
):
    with tasks(configurable=_configurable) as A, autoshard(
        _autoshard
    ) as B, strict_static_order(_strict_static_order) as C, enable_tracing(
        _enable_tracing
    ) as D, host_offload_min_reuse_distance(
        _host_offload_min_reuse_distance
    ) as E, only_fuse_loop_tasks(
        _only_fuse_loop_tasks
    ) as F, split_large_traces(
        _split_large_traces
    ) as G, enable_task_fusion(
        _enable_task_fusion
    ) as H, enable_fast_path(
        _enable_fast_path
    ) as J, ignore_transforms(
        _ignore_transforms
    ) as K:  # noqa: F841
        yield


def mjit(f, *args, in_shardings=None, out_shardings=None, **kwargs):
    # for any shardings that are named shardings, we should convert them into
    # logical names so we know how to translate logical names onto different
    # gpu submeshes within the computation
    def _annotate_shardings(x, sharding):
        if isinstance(sharding, NamedSharding):
            return with_sharding_constraint(x, sharding.spec)
        return x

    def wrapped(*fargs, **fkwargs):
        new_args = tree_map(_annotate_shardings, fargs, in_shardings)
        result = f(*new_args, **fkwargs)
        return tree_map(_annotate_shardings, result, out_shardings)

    return jax.jit(
        wrapped,
        *args,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        **kwargs,
    )


@contextmanager
def context(
    configurable=None,
    autoshard: Optional[bool] = None,
    strict_static_order: Optional[bool] = None,
    enable_tracing: Optional[bool] = None,
    host_offload_min_reuse_distance: Optional[int] = None,
    only_fuse_loop_tasks: Optional[bool] = None,
    split_large_traces: Optional[bool] = None,
    enable_task_fusion: Optional[bool] = None,
    enable_fast_path: Optional[bool] = None,
    ignore_transforms: Optional[bool] = None,
):
    with _context(
        _configurable=configurable,
        _autoshard=autoshard,
        _strict_static_order=strict_static_order,
        _enable_tracing=enable_tracing,
        _host_offload_min_reuse_distance=host_offload_min_reuse_distance,
        _only_fuse_loop_tasks=only_fuse_loop_tasks,
        _split_large_traces=split_large_traces,
        _enable_task_fusion=enable_task_fusion,
        _enable_fast_path=enable_fast_path,
        _ignore_transforms=ignore_transforms,
    ):
        yield
