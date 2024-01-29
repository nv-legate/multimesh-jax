import json
from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.ad_checkpoint import _optimization_barrier
from jax._src.lib import xla_client as xc
from jax.experimental.pjit import AUTO
from jax.lax import with_sharding_constraint
from jax.sharding import Mesh
from jax.tree_util import tree_map

from .lib import should_ignore_transforms
from .no_op import no_op

# colors 0 and 1 are reserved values
_next_color = 1
_task_depth = 0


class AutoParallelConcreteFxn:
    def __init__(self, compiled, in_shardings, out_shardings, init: Callable):
        self.compiled = compiled
        self.in_shardings = in_shardings
        self.out_shardings = out_shardings
        self.init = jax.jit(init, out_shardings=in_shardings)

    def init(self):
        return self.init()

    def __call__(self, args):
        return self.compiled(*args)


class AutoParallelAbstractFxn:
    def __init__(self, fxn, devices: np.ndarray):
        self.fxn = fxn
        self.flat_devices = np.asarray(devices).flatten()
        self.mesh = Mesh(self.flat_devices, ["x"])
        self.devices = devices

    def compile(self, init: Callable):
        arg_shapes = jax.eval_shape(init)
        abstract_args = jax.tree_map(
            lambda x: jax.core.ShapedArray(x.shape, x.dtype), arg_shapes
        )

        in_shardings = jax.tree_map(lambda x: AUTO(self.mesh), arg_shapes)
        result_shape = jax.eval_shape(self.fxn, *abstract_args)
        out_shardings = jax.tree_map(lambda x: AUTO(self.mesh), result_shape)

        jit_f = jax.jit(
            self.fxn, in_shardings=in_shardings, out_shardings=out_shardings
        )
        lowered = jit_f.lower(*abstract_args).compile()
        return AutoParallelConcreteFxn(
            lowered, lowered.input_shardings[0], lowered.output_shardings, init
        )


def parallelize(
    fxn, devices: Optional[Sequence[xc.Device] | np.ndarray] = None
):
    # the mesh is arbitrary and only serves to satisfy jax that a mesh exists
    # for the purposes of defining logical meshes that legate will remap
    # to physical meshes

    if devices is None:
        devices = jax.devices()

    return AutoParallelAbstractFxn(fxn, devices=devices)


def task(
    fxn,
    name: Optional[str] = None,
    *,
    counter=[0],
    out_shardings: Optional[Any] = None,
    devices: np.ndarray | Sequence[xc.Device] | None = None,
    device_axes: Optional[Sequence[str]] = None,
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None,
):
    global _next_color
    global _task_depth

    if should_ignore_transforms():
        return fxn

    if name is None:
        name = f"{fxn.__name__}.{counter[0]}"
        counter[0] += 1

    if devices is None:
        devices = [d.id for d in jax.devices()]
        dims = [len(devices)]
    elif isinstance(devices, np.ndarray):
        dims = devices.shape
        devices = [d.id for d in devices.flatten()]
    else:
        devices = [d.id for d in devices]
        dims = [len(devices)]

    if device_axes is not None and len(device_axes) != len(dims):
        raise ValueError(
            f"task {name}, device mesh with {len(dims)} dims does not match "
            f"device axies with {len(device_axes)} dims:  {device_axes}"
        )

    if device_axes is None:
        device_axes = []

    if logical_axes is None:
        logical_axes = []

    def _config_str(dependency_type: str, phase: str):
        global _next_color
        args = dict(
            type=dependency_type,
            name=f"{phase}.{name}",
            devices=devices,
            color=_next_color,
            autosharding=dict(
                dims=dims,
                device_axes=device_axes,
                logical_axes=logical_axes,
            ),
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


def abstract_microbatch(x):
    return x


def microbatch(
    fxn,
    dim: int,
    size: int,
    argnum: int = 0,
    num_stages: Optional[int] = None,
    max_breadth: Optional[int] = None,
):
    if should_ignore_transforms():
        return fxn

    def wrapped(*args, **kwargs):
        x = args[argnum]

        flat_x, _ = jax.tree_util.tree_flatten(x)
        microbatch_dim = flat_x[0].shape[dim]
        for fx in flat_x:
            if microbatch_dim != fx.shape[dim]:
                raise ValueError(
                    "dimensions not the same across all microbatched tensors"
                )

        num_microbatches = microbatch_dim // size
        if num_microbatches == 1:
            return fxn(*args, **kwargs)

        json_args = dict(
            num_stages=num_stages,
            max_breadth=max_breadth,
            num_microbatches=num_microbatches,
            slice_dim=dim,
            size=size,
        )

        mark_microbatch = no_op(
            name="Microbatch",
            abstract=abstract_microbatch,
            config=json.dumps(json_args),
        )
        mark_microbatch_slice = no_op(
            name="MicrobatchSlice",
            abstract=abstract_microbatch,
            config=json.dumps(json_args),
        )
        mark_microbatch_init = no_op(
            name="MicrobatchInit",
            abstract=abstract_microbatch,
            config=json.dumps(json_args),
        )

        def slice_microbatch(x, offset: int):
            offsets = [0] * len(x.shape)
            offsets[dim] = offset
            sizes = x.shape[:dim] + (size,) + x.shape[dim + 1 :]
            return mark_microbatch_slice(
                jax.lax.dynamic_slice(x, offsets, sizes)
            )

        abstract_slices = tree_map(lambda a: slice_microbatch(a, 0), x)

        new_args = args[:argnum] + (abstract_slices,) + args[argnum + 1 :]
        result_shapes = jax.eval_shape(fxn, *new_args, **kwargs)

        initial_results = tree_map(
            lambda x: jnp.zeros(x.shape, dtype=x.dtype), result_shapes
        )
        initial_results = tree_map(mark_microbatch_init, initial_results)

        x = tree_map(mark_microbatch, x)

        def body_fun(_, loop_args):
            (offset, prev_args) = loop_args

            slices = tree_map(lambda a: slice_microbatch(a, offset), x)

            # prep the offsets for the next loop
            offset += size
            flat_prev, treedef = jax.tree_util.tree_flatten(prev_args)
            new_args = args[:argnum] + (slices,) + args[argnum + 1 :]
            results = fxn(*new_args, **kwargs)
            flat_results, _ = jax.tree_util.tree_flatten(results)
            new_results = [x + y for x, y in zip(flat_prev, flat_results)]
            return (offset, jax.tree_util.tree_unflatten(treedef, new_results))

        offset = 0
        offset, result = jax.lax.fori_loop(
            0, num_microbatches, body_fun, (offset, initial_results)
        )
        return result

    return wrapped
