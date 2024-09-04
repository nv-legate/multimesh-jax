import json
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Type

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jax._src.lib import xla_client as xc
from jax.experimental.pjit import AUTO
from jax.interpreters import ad, mlir
from jax.interpreters.mlir import hlo, ir
from jax.sharding import Mesh, PartitionSpec as P
from jax.tree_util import tree_map

from .lib import (
    autoshard,
    optional_kwargs,
    should_ignore_transforms,
    with_sharding_constraint,
)
from .no_op import no_op

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
    model: Callable,
    *,
    init_params: Optional[Callable] = None,
    get_input_batch: Optional[Callable] = None,
    initial_batch: Optional[Any] = None,
    devices: Optional[Sequence[xc.Device] | np.ndarray] = None,
):
    if devices is None:
        devices = jax.devices()

    flat_devices = np.asarray(devices).flatten()
    mesh = Mesh(flat_devices, ["x"])

    with autoshard(True):
        param_shapes = jax.eval_shape(init_params)
        abstract_params = jax.tree_map(
            lambda x: jax.core.ShapedArray(x.shape, x.dtype), param_shapes
        )

        param_shardings = jax.tree_map(lambda x: AUTO(mesh), param_shapes)
        if initial_batch is None:
            if get_input_batch is None:
                raise ValueError(
                    "legate.jax.parallelize requires either get_input_batch"
                    "function or initial_batch parameter"
                )

            batch_shapes = jax.eval_shape(get_input_batch)
            batch_sharding = jax.tree_map(lambda x: AUTO(mesh), batch_shapes)
            initial_batch = jax.tree_map(
                lambda x: jax.core.ShapedArray(x.shape, x.dtype), batch_shapes
            )
        else:
            batch_sharding = jax.tree_map(lambda x: x.sharding, initial_batch)
            batch_shapes = jax.tree_map(
                lambda x: jax.core.ShapedArray(x.shape, x.dtype), initial_batch
            )

        result_shape = jax.eval_shape(model, param_shapes, batch_shapes)
        out_shardings = jax.tree_map(lambda x: AUTO(mesh), result_shape)

        jit_f = jax.jit(
            model,
            in_shardings=(param_shardings, batch_sharding),
            out_shardings=out_shardings,
        )

        compiled = jit_f.lower(abstract_params, initial_batch).compile()

    derived_param_shardings = compiled.input_shardings[0][0]
    derived_batch_shardings = compiled.input_shardings[0][1]
    init_sharded_params = jax.jit(
        init_params, out_shardings=derived_param_shardings
    )

    if get_input_batch is None:
        return compiled, init_sharded_params

    init_sharded_batch = jax.jit(
        get_input_batch, out_shardings=derived_batch_shardings
    )
    return compiled, init_sharded_params, init_sharded_batch


def put_to_devices(host_array: np.ndarray, devices) -> list[Any]:
    num_devices = len(devices)
    per_device_arrays = np.split(host_array, num_devices, axis=0)
    return jax.device_put(per_device_arrays, devices)


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


def __legate_task_lowering_impl(*args, jaxpr, **unused_kwargs):
    del unused_kwargs
    return jax_core.jaxpr_as_fun(jaxpr)(*args)


def _custom_abstract_eval(*args, jaxpr, **unused_kwargs):
    del unused_kwargs
    del args
    return jaxpr.out_avals


legate_task_p = jax_core.Primitive("legate_task")
legate_task_p.multiple_results = True
legate_task_p.def_abstract_eval(_custom_abstract_eval)
legate_task_p.def_impl(__legate_task_lowering_impl)


def call_legate_task(f, *args, config: str = "", **kwargs):
    jaxpr, out_shapes = jax.make_jaxpr(
        partial(f, **kwargs), return_shape=True
    )(*args)
    flat_args = jax.tree.leaves(args)
    out_tree = jax.tree.structure(out_shapes)
    out_flat = legate_task_p.bind(
        *flat_args, name=f.__name__, jaxpr=jaxpr, config=config
    )
    return jax.tree.unflatten(out_tree, out_flat)


def call_legate_task_fwd(f, *args, config: str = "", **kwargs):
    return call_legate_task(f, *args, config=config, **kwargs), args


def call_legate_task_bwd(f, primals, tangents, config: str = "", **kwargs):
    return call_legate_task(f, primals, tangents, config=config, **kwargs)


def _legate_task_lowering(
    ctx,
    *args,
    name,
    jaxpr,
    config: str = "",
):
    impl = mlir.core_call_lowering(
        ctx, *args, name=name + ".impl", call_jaxpr=jaxpr
    )
    call_op = impl[0].owner
    called_fn = call_op.attributes["callee"]
    legate_task = hlo.CustomCallOp(
        [r.type for r in call_op.results],
        call_op.operands,
        call_target_name="LegateTask",
        called_computations=ir.ArrayAttr.get([called_fn]),
        backend_config=ir.StringAttr.get(config),
    )
    return legate_task.results


mlir.register_lowering(legate_task_p, _legate_task_lowering)


def legate_task_linear(ct, _, **kwargs):
    return (legate_task_p.bind(ct, **kwargs),)


ad.deflinear2(legate_task_p, legate_task_linear)


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

    args = dict(
        name=name,
        devices=devices,
        autosharding=dict(
            dims=dims,
            device_axes=device_axes,
            logical_axes=logical_axes,
        ),
    )

    wrapped = partial(call_legate_task, fxn, config=json.dumps(args))

    fwd = partial(call_legate_task_fwd, fxn, config=json.dumps(args))

    def f_bwd(primals, tangents):
        _, f_vjp = jax.vjp(fxn, *primals)
        return f_vjp(tangents)

    bwd = partial(call_legate_task_bwd, f_bwd, config=json.dumps(args))

    vjp_taskify = jax.custom_vjp(wrapped)
    vjp_taskify.defvjp(fwd, bwd)
    return vjp_taskify


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
):
    if should_ignore_transforms():
        return fxn

    if name is None:
        name = f"{fxn.__name__}"
        counter[0] += 1

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
    interleave: Optional[int] = None,
    arg_shardings: Optional[Any] = None,
    schedule: Optional[str] = None,
    unrolling: Optional[int] = None,
    num_stages: Optional[int] = None,
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

        json_args = optional_kwargs(
            num_microbatches=num_microbatches,
            slice_dim=dim,
            size=size,
            batch_dim=dim,
            interleave=interleave,
            unrolling=unrolling,
            num_stages=num_stages,
            schedule=schedule,
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

        def slice_microbatch(x, offset: int, arg_pspec: P):
            if arg_pspec is not None:
                x = with_sharding_constraint(x, arg_pspec)

            sizes = x.shape[:dim] + (size,) + x.shape[dim + 1 :]
            offsets = [0] * len(x.shape)
            offsets[dim] = offset
            x = jax.lax.dynamic_slice(x, offsets, sizes)
            return mark_microbatch_slice(x)

        abstract_slices = tree_map(lambda a: slice_microbatch(a, 0, None), x)

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
                lambda a, arg_pspec: slice_microbatch(a, offset, arg_pspec),
                x,
                pre_slice_shardings,
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
