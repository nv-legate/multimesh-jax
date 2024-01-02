import json
from contextlib import contextmanager
from typing import Any

import jax
from jax._src.pjit import flatten_axis_resources
from jax.lax import with_sharding_constraint as lax_with_sharding_constraint
from jax.sharding import NamedSharding, PartitionSpec
from jax.tree_util import tree_flatten, tree_unflatten

from .legate_jax_impl import disable_implicit_tasks, enable_implicit_tasks
from .no_op import no_op

_ignore_transforms = 0


@contextmanager
def ignore_transforms(ignore: bool = True):
    global _ignore_transforms
    try:
        if ignore:
            if _ignore_transforms == 0:
                disable_implicit_tasks()
            _ignore_transforms += 1
        yield
    finally:
        if ignore:
            _ignore_transforms -= 1
            if _ignore_transforms == 0:
                enable_implicit_tasks()


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
            axes.append(pspec)
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
        mark_sharding = no_op(
            name="AutoSharding", config=json_str, abstract=lambda x: x
        )
        return mark_sharding(arg)

    flat_marked_args = [
        _mark_arg(pspec, arg) for pspec, arg in zip(axis_flat, flat_args)
    ]
    return tree_unflatten(arg_treedef, flat_marked_args)


def init(auto_shard: bool = False) -> None:
    if auto_shard:
        jax.lax.with_sharding_constraint = with_sharding_constraint
