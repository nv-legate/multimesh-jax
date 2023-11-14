import json
from typing import Any, Optional, Sequence

import jax
from jax._src.ad_checkpoint import _optimization_barrier
from jax.lax import with_sharding_constraint
from jax.tree_util import tree_map

from .lib import should_ignore_transforms
from .no_op import no_op

# color 0 is a reserved value
_next_color = 1
_task_depth = 0
_current_color = None


def task(
    fxn,
    name: Optional[str] = None,
    *,
    counter=[0],
    out_shardings: Optional[Any] = None,
    devices: Optional[Sequence[Any]] = None,
):
    global _next_color
    global _task_depth

    if should_ignore_transforms():
        return fxn

    if name is None:
        name = f"{fxn.__name__}.{counter[0]}"
        counter[0] += 1

    if devices is None:
        devices = []
    else:
        devices = [d.id for d in devices]

    def _config_str(dependency_type: str, phase: str):
        global _next_color
        args = dict(
            type=dependency_type,
            name=f"{phase}.{name}",
            devices=devices,
            color=_next_color,
        )
        return json.dumps(args)

    def start_task(inp):
        global _task_depth
        global _next_color
        _task_depth += 1
        # no nesting of tasks, only the outermost
        # task is actually carved out
        if _task_depth > 1:
            return inp

        _next_color += 1
        mark_input_fwd = no_op(
            name="TaskStart",
            config=_config_str("input", "fwd"),
            abstract=lambda x: x,
        )
        result = _optimization_barrier(tree_map(mark_input_fwd, inp))
        if out_shardings is not None:
            result = with_sharding_constraint(result, out_shardings)
        return result

    def finish_task(inp):
        global _task_depth
        _task_depth -= 1
        if _task_depth > 0:
            # no nesting of tasks, only the outermost
            # task is actually carved out
            return inp

        mark_output_fwd = no_op(
            name="TaskEnd",
            config=_config_str("output", "fwd"),
            abstract=lambda x: x,
        )
        return _optimization_barrier(tree_map(mark_output_fwd, inp))

    start = jax.custom_vjp(start_task)
    finish = jax.custom_vjp(finish_task)

    def args_task_barrier_fwd(inp):
        global _task_depth
        global _next_color
        _task_depth += 1
        # no nesting of tasks, only the outermost
        # task is actually carved out
        if _task_depth > 1:
            return inp, None

        _next_color += 1
        mark_input_fwd = no_op(
            name="TaskStart",
            config=_config_str("input", "fwd"),
            abstract=lambda x: x,
        )
        with jax.named_scope(f"args_{name}_forward"):
            result = _optimization_barrier(tree_map(mark_input_fwd, inp))
            if out_shardings is not None:
                result = with_sharding_constraint(result, out_shardings)
            return result, None

    def args_task_barrier_bwd(_, g):
        global _task_depth
        _task_depth -= 1
        if _task_depth > 0:
            # no nesting of tasks, only the outermost
            # task is actually carved out
            return (g,)

        mark_output_bwd = no_op(
            name="TaskEnd",
            config=_config_str("output", "bwd"),
            abstract=lambda x: x,
        )
        with jax.named_scope(f"args_{name}_backward"):
            return (_optimization_barrier(tree_map(mark_output_bwd, g)),)

    def result_task_barrier_fwd(inp):
        global _task_depth
        _task_depth -= 1
        # no nesting of tasks, only the outermost
        # task is actually carved out
        if _task_depth > 1:
            return inp, None

        mark_output_fwd = no_op(
            name="TaskEnd",
            config=_config_str("output", "fwd"),
            abstract=lambda x: x,
        )
        with jax.named_scope(f"result_{name}_forward"):
            return (
                _optimization_barrier(tree_map(mark_output_fwd, inp)),
                None,
            )

    def result_task_barrier_bwd(_, g):
        global _task_depth
        global _next_color
        _task_depth += 1
        # no nesting of tasks, only the outermost
        # task is actually carved out
        if _task_depth > 1:
            return (g,)

        _next_color += 1

        mark_input_bwd = no_op(
            name="TaskStart",
            config=_config_str("input", "bwd"),
            abstract=lambda x: x,
        )
        with jax.named_scope(f"result_{name}_backward"):
            return (_optimization_barrier(tree_map(mark_input_bwd, g)),)

    start.defvjp(args_task_barrier_fwd, args_task_barrier_bwd)
    finish.defvjp(result_task_barrier_fwd, result_task_barrier_bwd)

    def wrapped(*args):
        with jax.named_scope(f"task:{name}"):
            new_args = start(args)
            res = fxn(*new_args)
            return finish(res)

    return wrapped
