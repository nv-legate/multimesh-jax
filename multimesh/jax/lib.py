# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import contextmanager
from typing import Any, Callable, Optional, Sequence, Type, TypeAlias, Union

import jax
from jax._src.pjit import flatten_axis_resources
from jax.experimental.pjit import AUTO, pjit
from jax.lax import with_sharding_constraint as lax_with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.tree_util import tree_flatten, tree_unflatten

from .multimesh_jax_impl import (
    clear_tasks,
    enable_fast_path as _enable_fast_path,
    enable_metadata_name_tasks as _enable_metadata_name_tasks,
    enable_only_fuse_loop_tasks as _enable_only_fuse_loop_tasks,
    enable_recomputation as _enable_recomputation,
    enable_task_fusion as _enable_task_fusion,
)
from .no_op import no_op

_ignore_transforms = False

_auto_shard = False

ContextValue = Union[bool, int]


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
def ignore_transforms(ignore: Optional[bool] = True):
    """Sets debug context where all MultiMesh transformations are no-ops.

    To aid in debugging, functions can be invoked in a context in which
    all transformations like ``task`` and ``with_sharding_constraint``
    are ignored. This essentially falls back to default Jax execution.

    Args:
        ignore: optional, whether MultiMesh transformations should be ignored
    """
    global _ignore_transforms
    current = _ignore_transforms
    if ignore is None:
        yield
    else:
        try:
            if ignore:
                _enable_metadata_name_tasks(False)
                _ignore_transforms = True
            yield
        finally:
            _ignore_transforms = current
            if not current:
                _enable_metadata_name_tasks(True)


@contextmanager
def enable_fast_path(enable: Optional[bool] = None, context_value=[True]):
    """(en|dis)ables native Jax fallthrough path for jitted functions.

    For smaller utility functions, the MultiMesh overhead can be very high.
    The runtime can detect situations where bypassing MultiMesh is allowed
    and benefiical and falls through to a native CPU/GPU fast path.
    This can be disabled/re-enabled for debugging or performance tests.

    Args:
        enable: optional, whether to enable fast path fallthrough
        context_value: optional, a global variable holding the current
            context value. The user should never pass this value. The program
            begins in a context with ``enable`` True, which means that
            the fast path fallthrough is enabled.
    """
    with _set_context_value(enable, _enable_fast_path, context_value):
        yield


@contextmanager
def enable_task_fusion(enable: Optional[bool] = None, context_value=[True]):
    """Sets whether tasks should be allowed to fuse.

    Task fusion decreases runtime overhead by decreasing the number of tasks
    and creates extra opportunities for instruction-level fusion. Task fusion,
    however, decreases the total number of tasks and may reduce parallelism
    or increase the critical path length. This indicates whether tasks should
    be allowed to fuse.

    Args:
        enable: optional, whether to enable task fusion
        context_value: optional, a global variable holding the current
            context value. The user should never pass this value. The program
            begins in a context with ``enable`` True, which means that all
            tasks are allowed to fuse.
    """  # noqa: E501
    with _set_context_value(enable, _enable_task_fusion, context_value):
        yield


@contextmanager
def enable_recomputation(enable: Optional[bool] = None, context_value=[True]):
    """(dis|en)ables comm-avoiding recompute of inter-task intermediates.

    Inter-task intermediates create extra runtime overhead and may
    cause extra inter-task communication. This recomputation can be
    enabled or disabled for debugging or performance tests.
    This recomputation is separate and complementary to Jax-level
    checkpointing and HLO remateralization passes.

    Args:
        enable: optional, whether to enable inter-task recomputation
        context_value: optional, a global variable holding the current
            context value. The user should never pass this value. The program
            begins in a context with ``enable`` True, which means that
            recomputation is enabled.
    """
    with _set_context_value(enable, _enable_recomputation, context_value):
        yield


@contextmanager
def only_fuse_loop_tasks(enable: Optional[bool] = None, context_value=[False]):
    """Sets whether tasks outside microbatch loops should be allowed to fuse.

    Task fusion decreases runtime overhead by decreasing the number of tasks
    and creates extra opportunities for instruction-level fusion. Task fusion,
    however, decreases the total number of tasks and may reduce parallelism
    or increase the critical path length. This specifically indicates that
    tasks inside a loop (which are repeatedly executed) should be fused
    to minimize overhead and maximize fusions.

    Args:
        enable: optional, whether only tasks inside a loop should be
            allowed to fuse
        context_value: optional, a global variable holding the current context
            value. The user should never pass this value. The program begins
            in a context with ``enable`` False, which means that all tasks are
            allowed to fuse.
    """
    with _set_context_value(
        enable, _enable_only_fuse_loop_tasks, context_value
    ):
        yield


@contextmanager
def autoshard(autoshard: bool = True):
    """Opens a context where shardings constraints become logical shardings.

    Args:
        autoshard: optional, whether the context should use logical
         autosharding.
    """
    global _auto_shard
    if autoshard is not None:
        current = _auto_shard
        _auto_shard = autoshard
    yield
    if autoshard is not None:
        _auto_shard = current


def should_ignore_transforms() -> bool:
    backend = jax._src.xla_bridge.get_backend()
    if _ignore_transforms:
        return True

    backend = jax._src.xla_bridge.get_backend()
    return backend.platform != "multimesh"


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
    """Applies logical sharding annotations to the input(s)

    Matches the semantics of `jax.lax.with_sharding_constraint`_.
    The shardings (``axis_resources``) should be ``PartitionSpec``
    or ``NamedSharding`` objects that have named axes. If no
    autosharding context is active, this forwards to
    `jax.lax.with_sharding_constraint`_.

    Args:
      x: array or pytree of arrays to apply shardings to
      axis_resources: sharding or pytree of shardings. This should be a
        `PartitionSpec` or `NamedSharding` or tuple of axis names.

    Returns:
      The input `x` with sharding constraints.

    Example:
      >>> from jax_plugins.multimesh import init
      >>> init(cpus=4)
      >>>
      >>> import jax
      >>> import jax.numpy as jnp
      >>> import numpy as np
      >>> from jax.sharding import PartitionSpec as P, Mesh
      >>> from jax.experimental.pjit import pjit
      >>> from multimesh.jax import with_sharding_constraint, task, autoshard
      >>>
      >>> mesh = Mesh(np.array(jax.devices()), ("x",))
      >>> def f():
      ...   x = jnp.arange(16)
      ...   return with_sharding_constraint(x, P("x"))
      >>>
      >>> jit_f = pjit(task(f, mesh=mesh))
      >>> with mesh, autoshard(True):
      ...   y = jit_f()
      >>> jax.debug.visualize_array_sharding(y)
        ┌────────┬────────┬────────┬────────┐
        │LEGATE 0│LEGATE 1│LEGATE 2│LEGATE 3│
        └────────┴────────┴────────┴────────┘
    .. _jax.lax.with_sharding_constraint: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.with_sharding_constraint.html
    """  # noqa: E501
    global _auto_shard

    if _ignore_transforms or not _auto_shard:
        return lax_with_sharding_constraint(x, axis_resources)
    flat_args, arg_treedef = tree_flatten(x)
    axis_flat = flatten_axis_resources(
        "multimesh-jax", arg_treedef, axis_resources, tupled_args=True
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


@contextmanager
def tasks(configurable: Optional[Type] = None):
    if configurable is not None:
        clear_tasks()
        task_configure = configurable()
        task_configure()

    yield

    if configurable is not None:
        clear_tasks()


def mjit(
    f,
    *args,
    mesh=None,
    devices=None,
    in_shardings=None,
    out_shardings=None,
    **kwargs,
):
    # for any shardings that are named shardings, we should convert them into
    # logical names so we know how to translate logical names onto different
    # gpu submeshes within the computation
    def _annotate_shardings(x, sharding):
        if isinstance(sharding, NamedSharding):
            return with_sharding_constraint(x, sharding.spec)
        return x

    def wrapped(*fargs, **fkwargs):
        if in_shardings is not None:
            new_args = jax.tree.map(_annotate_shardings, fargs, in_shardings)
        else:
            new_args = fargs
        result = f(*new_args, **fkwargs)
        if out_shardings is not None:
            return jax.tree.map(_annotate_shardings, result, out_shardings)
        return result

    if out_shardings is None:
        if mesh is None:
            if devices is None:
                raise ValueError(
                    "one of out_shardings, mesh, or devices must be passed to mjit"  # noqa: E501
                )
            mesh = Mesh(devices, ("x",))
        out_shardings = AUTO(mesh)

    return pjit(
        wrapped,
        *args,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        **kwargs,
    )


@contextmanager
def _context(
    _configurable=None,
    _autoshard: Optional[bool] = None,
    _enable_recomputation: Optional[bool] = None,
    _only_fuse_loop_tasks=None,
    _enable_task_fusion=None,
    _enable_fast_path=None,
    _ignore_transforms=None,
):
    with tasks(configurable=_configurable) as A, autoshard(
        _autoshard
    ) as B, only_fuse_loop_tasks(
        _only_fuse_loop_tasks
    ) as C, enable_task_fusion(
        _enable_task_fusion
    ) as D, enable_fast_path(
        _enable_fast_path
    ) as E, ignore_transforms(
        _ignore_transforms
    ) as F, enable_recomputation(
        _enable_recomputation
    ):  # noqa: F841
        yield


@contextmanager
def context(
    autoshard: Optional[bool] = None,
    enable_recomputation: Optional[bool] = None,
    only_fuse_loop_tasks: Optional[bool] = None,
    enable_task_fusion: Optional[bool] = None,
    enable_fast_path: Optional[bool] = None,
    ignore_transforms: Optional[bool] = None,
):
    """Helper function to configure multiple contexts in a single manager.

    Args:
        autoshard: optional, value to configure the :func:`.autoshard` context manager.
        only_fuse_loop_tasks: optional, value to configure the :func:`.only_fuse_loop_tasks` context manager.
        enable_task_fusion: optional, value to configure the :func:`.enable_task_fusion` context manager.
        enable_fast_path: optional, value to configure the :func:`.enable_fast_path` context manager.
        ignore_transforms: optional, value to configure the :func:`.ignore_transforms` context manager.
    """  # noqa: E501
    with _context(
        _autoshard=autoshard,
        _enable_recomputation=enable_recomputation,
        _only_fuse_loop_tasks=only_fuse_loop_tasks,
        _enable_task_fusion=enable_task_fusion,
        _enable_fast_path=enable_fast_path,
        _ignore_transforms=ignore_transforms,
    ):
        yield
