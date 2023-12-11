import json
from contextlib import contextmanager
from typing import Any

import jax
from jax._src.pjit import flatten_axis_resources
from jax.tree_util import tree_flatten, tree_unflatten

from .no_op import no_op

_ignore_transforms = 0


@contextmanager
def ignore_transforms(ignore: bool = True):
    global _ignore_transforms
    if ignore:
        _ignore_transforms += 1
    yield
    if ignore:
        _ignore_transforms -= 1


def should_ignore_transforms() -> bool:
    backend = jax._src.xla_bridge.get_backend()
    if _ignore_transforms > 0:
        return True

    backend = jax._src.xla_bridge.get_backend()
    return backend.platform != "legate"


def with_sharding_constraint_wrapper(x: Any, axis_resources: Any):
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
                "axes": list(pspec.spec),
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
        jax.lax.with_sharding_constraint = with_sharding_constraint_wrapper
