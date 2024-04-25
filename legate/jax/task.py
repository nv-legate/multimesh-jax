import json
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Type

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax._src.ad_checkpoint import _optimization_barrier
from jax._src.lib import xla_client as xc
from jax.experimental.pjit import AUTO, pjit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.tree_util import tree_map

from .lib import (
    optional_kwargs,
    should_ignore_transforms,
    with_sharding_constraint,
)
from .no_op import no_op

_next_color = 0
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
    if devices is None:
        devices = jax.devices()

    return AutoParallelAbstractFxn(fxn, devices=devices)


def put_to_devices(host_array: np.ndarray, devices) -> list[Any]:
    num_devices = len(devices)
    per_device_arrays = np.split(host_array, num_devices, axis=0)
    return jax.device_put(per_device_arrays, devices)


def parallelize_step(
    model,
    optimizer,
    batch: Any,
    mesh: Optional[Mesh] = None,
    fully_shard_first_batch_dim: bool = True,
):
    import flax.linen as nn
    from flax.training.train_state import TrainState

    if mesh is None:
        mesh = Mesh(jax.devices(), ("x",))

    def init_fn(k, x, model, optimizer):
        params = model.init(k, x)
        state = TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        )
        return state

    init_fn = partial(init_fn, model=model, optimizer=optimizer)

    variable_avals = jax.eval_shape(init_fn, random.key(42), batch)
    variable_spec = nn.get_partition_spec(variable_avals)
    grad_fn = jax.value_and_grad(model.apply)

    def label_sharding(x, s):
        return jax.lax.with_sharding_constraint(x, s)

    def step_fn(variables, inputs):
        flat_vars, treedef = jax.tree_util.tree_flatten(variables)
        flat_spec, _ = jax.tree_util.tree_flatten(variable_spec)
        flat_vars = [
            label_sharding(v, s) for v, s in zip(flat_vars, flat_spec)
        ]
        variables = jax.tree_util.tree_unflatten(treedef, flat_vars)
        params = variables.params
        loss, grads = grad_fn(params, inputs)
        variables = variables.apply_gradients(grads=grads)
        return loss, variables

    variable_shardings = jax.tree_map(lambda x: AUTO(mesh), variable_avals)

    result_shape = jax.eval_shape(step_fn, variable_avals, batch)
    out_shardings = jax.tree_map(lambda x: AUTO(mesh), result_shape)

    if fully_shard_first_batch_dim:
        first_axis_name = mesh.axis_names[0]
        batch_shardings = jax.tree_map(
            lambda x: NamedSharding(mesh, P(first_axis_name)), batch
        )
    else:
        batch_shardings = jax.tree_map(lambda x: AUTO(mesh), batch)
    batch_avals = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, dtype=x.dtype), batch
    )

    # pjit is required here so we get the global mesh context
    compiled_step = (
        pjit(
            step_fn,
            in_shardings=(variable_shardings, batch_shardings),
            out_shardings=out_shardings,
        )
        .lower(variable_avals, batch_avals)
        .compile()
    )
    variable_shardings, batch_shardings = compiled_step.input_shardings[0]

    def make_sharded_array(host_array, sharding):
        device_buffers = put_to_devices(host_array, mesh.local_devices)
        return jax.make_array_from_single_device_arrays(
            host_array.shape, sharding, device_buffers
        )

    def prepare_batch(replicated_batch_arrays):
        return jax.tree_map(
            make_sharded_array, replicated_batch_arrays, batch_shardings
        )

    init_fn = pjit(init_fn, out_shardings=variable_shardings)
    init_variables = init_fn(random.key(42), batch)

    return compiled_step, init_variables, prepare_batch, mesh


mark_output = no_op(name="TaskEnd")
mark_input = no_op(name="TaskStart")


def shard_axes(*args):
    import flax.linen as nn

    return nn.with_partitioning(nn.initializers.xavier_normal(), args)


@contextmanager
def check_nested_task():
    global _task_depth
    current_depth = _task_depth
    _task_depth += 1

    yield current_depth

    _task_depth -= 1


@dataclass
class Task:
    mesh: Optional[Mesh] = None
    devices: np.ndarray | Sequence[xc.Device] | None = None
    device_axes: Optional[Sequence[str]] = None
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None


def _get_wrapped_task(
    fxn,
    name: str,
    *,
    out_shardings: Optional[Any] = None,
    devices: np.ndarray | Sequence[xc.Device] | None = None,
    mesh: Optional[Mesh] = None,
    device_axes: Optional[Sequence[str]] = None,
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None,
):
    if mesh is not None:
        if devices is not None or device_axes is not None:
            raise ValueError(
                "cannot give both mesh and "
                "devices/device_axes arguments to Legate task"
            )
        dims = mesh.shape
        devices = mesh.devices
        device_axes = mesh.axis_names
        if logical_axes is None:
            logical_axes = [(ax, ax) for ax in device_axes]

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
        global _next_color

        _next_color += 1

        mark_input_fwd = partial(
            mark_input, config=_config_str("input", "fwd")
        )
        result = _optimization_barrier(tree_map(mark_input_fwd, inp))
        if out_shardings is not None:
            result = jax.lax.with_sharding_constraint(result, out_shardings)
        return result

    def finish_task(inp):
        mark_output_fwd = partial(
            mark_output, config=_config_str("output", "fwd")
        )
        return _optimization_barrier(tree_map(mark_output_fwd, inp))

    start = jax.custom_vjp(start_task)
    finish = jax.custom_vjp(finish_task)

    def args_task_barrier_fwd(inp):
        global _next_color

        _next_color += 1
        mark_input_fwd = partial(
            mark_input, config=_config_str("input", "fwd")
        )
        with jax.named_scope(f"args_{name}_forward"):
            result = _optimization_barrier(tree_map(mark_input_fwd, inp))
            if out_shardings is not None:
                result = jax.lax.with_sharding_constraint(
                    result, out_shardings
                )
            return result, None

    def args_task_barrier_bwd(_, g):
        mark_output_bwd = partial(
            mark_output, config=_config_str("output", "bwd")
        )
        with jax.named_scope(f"args_{name}_backward"):
            return (_optimization_barrier(tree_map(mark_output_bwd, g)),)

    def result_task_barrier_fwd(inp):
        mark_output_fwd = partial(
            mark_output, config=_config_str("output", "fwd")
        )
        with jax.named_scope(f"result_{name}_forward"):
            return (
                _optimization_barrier(tree_map(mark_output_fwd, inp)),
                None,
            )

    def result_task_barrier_bwd(_, g):
        global _next_color

        _next_color += 1

        mark_input_bwd = partial(
            mark_input, config=_config_str("input", "bwd")
        )
        with jax.named_scope(f"result_{name}_backward"):
            return (_optimization_barrier(tree_map(mark_input_bwd, g)),)

    start.defvjp(args_task_barrier_fwd, args_task_barrier_bwd)
    finish.defvjp(result_task_barrier_fwd, result_task_barrier_bwd)

    def wrapped(*args, **kwargs):
        with jax.named_scope(f"task:{name}"):
            with check_nested_task() as depth:
                if depth > 0:
                    # for now only the outermost task matters
                    return fxn(*args, **kwargs)

                new_args = start(args)
                res = fxn(*new_args, **kwargs)
                return finish(res)

    return wrapped


def task(
    fxn: Callable | Type,
    name: Optional[str] = None,
    *,
    counter=[0],
    out_shardings: Optional[Any] = None,
    mesh: Optional[Mesh] = None,
    devices: np.ndarray | Sequence[xc.Device] | None = None,
    device_axes: Optional[Sequence[str]] = None,
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None,
    configure: Optional[Callable[..., Task]] = None,
    configure_args: Optional[Sequence[str]] = None,
):
    global _next_color
    global _task_depth

    if should_ignore_transforms():
        return fxn

    if isinstance(fxn, type):
        # we need to transform functions, not types, which means
        # we neee to "defer" the transformation until this class is
        # instantiated
        class WrappedTask(fxn):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                parent_call = super().__call__
                wrapped_call = task(
                    parent_call,
                    name,
                    out_shardings=out_shardings,
                    mesh=mesh,
                    devices=devices,
                    device_axes=device_axes,
                    logical_axes=logical_axes,
                    configure=configure,
                    configure_args=configure_args,
                )
                return wrapped_call(*args, **kwargs)

        return WrappedTask

    if name is None:
        name = f"{fxn.__name__}.{counter[0]}"
        counter[0] += 1

    if configure is not None:
        if (
            devices is not None
            or device_axes is not None
            or logical_axes is not None
        ):
            raise ValueError(
                "cannot give both a configure type and devices/device_axes"
                "/logical_axees to legate.jax.task"
            )

        if out_shardings is not None:
            raise ValueError(
                "cannot give both configure and out_shardings to task"
            )

        if configure_args:
            # we can't know all the arguments until the function is invoked
            # so we have to defer creating the actual wrapped task until the
            # function is called
            def wrapped(*args, **kwargs):
                configure_kwargs = {}
                if configure_args is not None:
                    for arg in configure_args:
                        configure_kwargs[arg] = kwargs[arg]

                task_config: Task = configure(**configure_kwargs)

                return _get_wrapped_task(
                    fxn,
                    name,
                    out_shardings=out_shardings,
                    mesh=task_config.mesh,
                    devices=task_config.devices,
                    device_axes=task_config.device_axes,
                    logical_axes=task_config.logical_axes,
                )(*args, **kwargs)

            return wrapped

        task_config: Task = configure()
        devices = task_config.devices
        device_axes = task_config.device_axes
        logical_axes = task_config.logical_axes

    return _get_wrapped_task(
        fxn,
        name,
        mesh=mesh,
        out_shardings=out_shardings,
        devices=devices,
        device_axes=device_axes,
        logical_axes=logical_axes,
    )


def abstract_microbatch(x):
    return x


def microbatch(
    fxn,
    dim: int,
    size: int,
    argnum: int = 0,
    interleave: int = 1,
    batch_reshape: Optional[int] = None,
    arg_shardings: Optional[Any] = None,
    microbatch_shardings: Optional[Any] = None,
    unrolling: Optional[int] = None,
    num_pipeline_stages: Optional[int] = None,
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

        if batch_reshape is not None:
            slice_dim = dim + 1
            slice_size = size // batch_reshape
        else:
            slice_dim = dim
            slice_size = size

        json_args = optional_kwargs(
            num_microbatches=num_microbatches,
            slice_dim=slice_dim,
            size=slice_size,
            batch_dim=dim,
            interleave=interleave,
            unrolling=unrolling,
            num_pipeline_stages=num_pipeline_stages,
        )

        mark_microbatch = no_op(
            name="Microbatch",
            config=json.dumps(json_args),
        )
        mark_microbatch_slice = no_op(
            name="MicrobatchSlice",
            config=json.dumps(json_args),
        )
        mark_microbatch_init = no_op(
            name="MicrobatchInit",
            config=json.dumps(json_args),
        )

        if arg_shardings is None or isinstance(arg_shardings, P):
            pre_slice_shardings = jax.tree_map(lambda a: arg_shardings, x)
        else:
            pre_slice_shardings = arg_shardings

        if microbatch_shardings is None or isinstance(microbatch_shardings, P):
            post_slice_shardings = jax.tree_map(
                lambda a: microbatch_shardings, x
            )
        else:
            post_slice_shardings = microbatch_shardings

        def slice_microbatch(x, offset: int, arg_pspec: P, slice_pspec: P):
            if arg_pspec is not None:
                x = with_sharding_constraint(x, arg_pspec)

            if batch_reshape is None:
                offsets = [0] * len(x.shape)
                offsets[dim] = offset
                sizes = x.shape[:dim] + (size,) + x.shape[dim + 1 :]
                x = jax.lax.dynamic_slice(x, offsets, sizes)
            else:
                reshaped = (
                    x.shape[:dim]
                    + (batch_reshape, x.shape[dim] // batch_reshape)
                    + x.shape[dim + 1 :]
                )
                final_shape = x.shape[:dim] + (size,) + x.shape[dim + 1 :]
                sizes = (
                    x.shape[:dim]
                    + (batch_reshape, size // batch_reshape)
                    + x.shape[dim + 1 :]
                )
                offsets = [0] * (len(x.shape) + 1)
                offsets[dim + 1] = offset // batch_reshape
                x = x.reshape(reshaped)
                x = jax.lax.dynamic_slice(x, offsets, sizes)
                x = x.reshape(final_shape)

            if slice_pspec is not None:
                x = with_sharding_constraint(x, slice_pspec)
            return mark_microbatch_slice(x)

        abstract_slices = tree_map(
            lambda a: slice_microbatch(a, 0, None, None), x
        )

        new_args = args[:argnum] + (abstract_slices,) + args[argnum + 1 :]
        result_shapes = jax.eval_shape(fxn, *new_args, **kwargs)

        initial_results = tree_map(
            lambda x: jnp.zeros(x.shape, dtype=x.dtype), result_shapes
        )
        initial_results = tree_map(mark_microbatch_init, initial_results)

        x = tree_map(mark_microbatch, x)

        def body_fun(_, loop_args):
            (offset, prev_args) = loop_args

            slices = tree_map(
                lambda a, arg_pspec, slice_pspec: slice_microbatch(
                    a, offset, arg_pspec, slice_pspec
                ),
                x,
                pre_slice_shardings,
                post_slice_shardings,
            )

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
