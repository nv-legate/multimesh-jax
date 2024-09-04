from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Type

import jax
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import random
from jax._src.lib import xla_client as xc
from jax.experimental.pjit import AUTO, pjit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..lib import autoshard, should_ignore_transforms
from ..task import put_to_devices, task as functional_task


def task(
    fxn: Callable | Type,
    name: Optional[str] = None,
    *,
    out_shardings: Optional[Any] = None,
    mesh: Optional[Mesh] = None,
    devices: np.ndarray | Sequence[xc.Device] | None = None,
    device_axes: Optional[Sequence[str]] = None,
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None,
):
    if should_ignore_transforms():
        return fxn

    class ManualTask(nn.Module):
        @nn.compact
        def __call__(self, xs):
            mod = fxn(parent=None)
            init_fn = mod.init
            apply_fn = mod.apply
            apply_fn = functional_task(
                mod.apply,
                name,
                out_shardings=out_shardings,
                mesh=mesh,
                devices=devices,
                device_axes=device_axes,
                logical_axes=logical_axes,
            )
            mod_params = self.param("mod", init_fn, xs)
            return apply_fn(mod_params, xs)

    return ManualTask


def shard_axes(*args):
    return nn.with_partitioning(nn.initializers.xavier_normal(), args)


def parallelize_step(
    model,
    optimizer,
    batch: Any,
    mesh: Optional[Mesh] = None,
    fully_shard_first_batch_dim: bool = True,
):
    if mesh is None:
        mesh = Mesh(jax.devices(), ("x",))

    def init_fn(k, x, model, optimizer):
        params = model.init(k, x)
        state = TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        )
        return state

    init_fn = partial(init_fn, model=model, optimizer=optimizer)

    with autoshard(True):
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

    with autoshard(True):
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
    with mesh, autoshard(True):
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

    with mesh, autoshard(True):
        init_fn = pjit(init_fn, out_shardings=variable_shardings)
        init_variables = init_fn(random.key(42), batch)

    return compiled_step, init_variables, prepare_batch, mesh
