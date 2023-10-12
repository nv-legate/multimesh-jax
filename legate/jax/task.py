from typing import Optional

import jax
from jax._src.ad_checkpoint import _optimization_barrier

from .no_op import no_op


def start_task(inp):
    return _optimization_barrier(inp)


def finish_task(inp):
    return _optimization_barrier(inp)


def task(fxn, name: Optional[str] = None, counter=[0]):
    if name is None:
        name = f"{fxn.__name__}.{counter[0]}"
        counter[0] += 1

    mark_input_fwd = no_op(
        name="Task", config=f"input:{name}.fwd", abstract=lambda x: x
    )
    mark_output_fwd = no_op(
        name="Task", config=f"output:{name}.fwd", abstract=lambda x: x
    )
    mark_input_bwd = no_op(
        name="Task", config=f"input:{name}.bwd", abstract=lambda x: x
    )
    mark_output_bwd = no_op(
        name="Task", config=f"output:{name}.bwd", abstract=lambda x: x
    )

    start = jax.custom_vjp(start_task)
    finish = jax.custom_vjp(finish_task)

    def args_task_barrier_fwd(inp):
        with jax.named_scope(f"args_{name}_forward"):
            return (
                _optimization_barrier(
                    jax.tree_util.tree_map(mark_input_fwd, inp)
                ),
                None,
            )

    def args_task_barrier_bwd(_, g):
        with jax.named_scope(f"args_{name}_backward"):
            return (
                _optimization_barrier(
                    jax.tree_util.tree_map(mark_output_bwd, g)
                ),
            )

    def result_task_barrier_fwd(inp):
        with jax.named_scope(f"result_{name}_forward"):
            return (
                _optimization_barrier(
                    jax.tree_util.tree_map(mark_output_fwd, inp)
                ),
                None,
            )

    def result_task_barrier_bwd(_, g):
        with jax.named_scope(f"result_{name}_backward"):
            return (
                _optimization_barrier(
                    jax.tree_util.tree_map(mark_input_bwd, g)
                ),
            )

    start.defvjp(args_task_barrier_fwd, args_task_barrier_bwd)
    finish.defvjp(result_task_barrier_fwd, result_task_barrier_bwd)

    def wrapped(*args):
        with jax.named_scope(f"task:{name}"):
            new_args = start(args)
            res = fxn(*new_args)
            return finish(res)

    return wrapped
