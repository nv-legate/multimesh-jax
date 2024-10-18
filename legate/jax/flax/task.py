from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple, Type

import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import random
from jax._src.lib import xla_client as xc
from jax.experimental.pjit import AUTO, pjit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..lib import autoshard, should_ignore_transforms
from ..task import task as pure_function_task, with_sharding_constraint


def _put_to_devices(host_array: np.ndarray, devices) -> list[Any]:
    num_devices = len(devices)
    per_device_arrays = np.split(host_array, num_devices, axis=0)
    return jax.device_put(per_device_arrays, devices)


def task(
    model: Type[nn.Module],
    name: Optional[str] = None,
    *,
    mesh: Optional[Mesh] = None,
    devices: np.ndarray | Sequence[xc.Device] | None = None,
    device_axes: Optional[Sequence[str]] = None,
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None,
):
    """Wraps a Flax module in an auto-sharding task context

    Args:
      model: Flax model to be encapsulated as a task.
      name: optional, a metadata name to assign to the task context
      mesh: optional, a Mesh context defining the devices and mesh shape
      devices: optional, a numpy array or list of jax devices
        specifying the devices to include in the task submesh.
        One of ``mesh`` or ``devices`` must be given. If ``devices``
        is a numpy array, the mesh shape is inferred from the shape
        of the device array.
      device_axes: optional, a list of names to assign to each device axis.
        The number of names must match the shape of ``mesh`` or ``devices``.
        If None, logical sharding constraints will be translated
        to replicated sharding.
      logical_axes: optional, a list of string pairs ('logical', 'device')
        giving the translation from logical names to physical device names.
        The logical names should match those passed to
        ``with_sharding_constraint`` calls within the task. If None,
        the device axis names are used directly for autosharding.
        Raises a ``ValueError`` if ``logical_axes`` are given
        but no  ``mesh`` or ``device_axes`` are specified.

    Returns:
      A wrapped version of ``model`` usable as a submesh task.

    Example:
      >>> import numpy as np
      >>> import jax
      >>> from legate.jax.flax import shard_axes, task
      >>> from flax import linen as nn
      >>> from jax.sharding import Mesh
      >>>
      >>> devices = np.array(jax.devices()).reshape(2,2)
      >>> mesh = Mesh(devices, ("x", "y"))
      >>> DenseTask = task(nn.Dense, mesh=mesh)
      >>> model = DenseTask(features=16, kernel_init=shard_axes("x", None))
      >>>
      >>> x =  jnp.ones((16,9))
      >>> model.tabulate(jax.random.key(0), x))

    Which produces the output::

                                                Task[Dense] Summary
        ┏━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ path ┃ module      ┃ inputs        ┃ outputs        ┃ params                               ┃
        ┡━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │      │ Task[Dense] │ float32[16,9] │ float32[16,16] │ mod:                                 │
        │      │             │               │                │   params:                            │
        │      │             │               │                │     bias: float32[16]                │
        │      │             │               │                │     kernel: float32[9,16] P(x, None) │
        │      │             │               │                │                                      │
        │      │             │               │                │ 160 (640 B)                          │
        ├──────┼─────────────┼───────────────┼────────────────┼──────────────────────────────────────┤
        │      │             │               │          Total │ 160 (640 B)                          │
        └──────┴─────────────┴───────────────┴────────────────┴──────────────────────────────────────┘

                                        Total Parameters: 160 (640 B)

    """  # noqa: E501
    if should_ignore_transforms():
        return model

    class ManualTask(model):
        def __init__(self, *args, **kwargs):
            self.mod = model(*args, **kwargs)
            super().__init__(*args, **kwargs)

        def __post_init__(self):
            super().__post_init__()

        @nn.compact
        def __call__(self, xs):
            init_fn = self.mod.init
            apply_fn = self.mod.apply
            apply_fn = pure_function_task(
                apply_fn,
                name,
                mesh=mesh,
                devices=devices,
                device_axes=device_axes,
                logical_axes=logical_axes,
            )
            mod_params = self.param("mod", init_fn, xs)
            return apply_fn(mod_params, xs)

    ManualTask.__name__ = f"Task[{model.__name__}]"
    return ManualTask


def shard_axes(*args: str) -> Callable:
    """Helper function to simplify naming axes of Flax parameters

    Args:
      *args: Sequence of axis names to apply to a Flax kernel

    Returns:
      A partitioning function that will apply logical names
      to the axes of module parameters

    Example:
      >>> import numpy as np
      >>> import jax
      >>> from legate.jax.flax import shard_axes
      >>> from flax import linen as nn
      >>> from jax.sharding import Mesh
      >>>
      >>> model = Dense(features=16, kernel_init=shard_axes("batch", "model"))
      >>>
      >>> x =  jnp.ones((16,9))
      >>> model.tabulate(jax.random.key(0), x))

    Which produces the output::

                                        Dense Summary
        ┏━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ path ┃ module ┃ inputs        ┃ outputs        ┃ params                                ┃
        ┡━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │      │ Dense  │ float32[16,9] │ float32[16,16] │ bias: float32[16]                     │
        │      │        │               │                │ kernel: float32[9,16] P(batch, model) │
        │      │        │               │                │                                       │
        │      │        │               │                │ 160 (640 B)                           │
        ├──────┼────────┼───────────────┼────────────────┼───────────────────────────────────────┤
        │      │        │               │          Total │ 160 (640 B)                           │
        └──────┴────────┴───────────────┴────────────────┴───────────────────────────────────────┘

                                    Total Parameters: 160 (640 B)

    """  # noqa: E501
    return nn.with_partitioning(nn.initializers.xavier_normal(), args)


def parallelize_step(
    model: Type[nn.Module],
    optimizer: optax.GradientTransformation,
    mesh: Mesh,
    global_batch: Optional[Any] = None,
    local_batch: Optional[Any] = None,
    replicate_inputs: bool = True,
):
    """Combines model loss function and optimizer into an autosharded step function

    Args:
      model: Flax model to automatically convert to a full training step
      optimizer: The optimizer to use for the parameters
      mesh: a Mesh context defining the devices and mesh shape
      global_batch: optional, an example batch giving the full (global) input shapes.  This
        must be an array with global sharding information.
      local_batch: optional, an example per-process batch giving the local input shape.

    Returns:
      If ``global_batch`` is given, returns a tuple of (``sharded_model``, ``initial_params``, ``mesh``).
      If ``local_batch`` is given, returns a tuple of (``sharded_model``, ``initial_parmas``, ``mesh``, ``prepare_batch``).
      The returned ``initial_params`` contains fully initialized, sharded parameters.
      The ``mesh`` is a mesh context suitable for MPMD invocations of ``sharded_model``.
      The ``prepare_batch`` converts a local per-process batch into a global, sharded array.

    Example:
      See the tutorial notebooks for complete examples.
    """  # noqa: E501

    if mesh is None:
        mesh = Mesh(jax.devices(), ("x",))

    def init_fn(k, x, model, optimizer):
        params = model.init(k, x)
        state = TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        )
        return state

    init_fn = partial(init_fn, model=model, optimizer=optimizer)

    num_data_loaders = mesh.devices.size // len(mesh.local_devices)
    if global_batch is not None:
        batch_shardings = jax.tree_map(lambda x: x.sharding, global_batch)
        batch_avals = jax.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, dtype=x.dtype),
            global_batch,
        )

    elif local_batch is not None:
        first_axis_name = mesh.axis_names[0]

        def get_global_batch(x):
            if len(x.shape) == 0:
                return jax.ShapeDtypeStruct(x.shape, dtype=x.dtype)

            shape = (x.shape[0] * num_data_loaders,) + x.shape[1:]
            return jax.ShapeDtypeStruct(shape, dtype=x.dtype)

        batch_avals = jax.tree_map(get_global_batch, local_batch)
        batch_shardings = jax.tree_map(
            lambda x: NamedSharding(mesh, P(first_axis_name)), local_batch
        )

    with autoshard(True):
        variable_avals = jax.eval_shape(init_fn, random.key(42), batch_avals)
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
        if replicate_inputs:
            inputs = jax.tree.map(lambda x: with_sharding_constraint(x, P()), inputs)
        variables = jax.tree_util.tree_unflatten(treedef, flat_vars)
        params = variables.params
        loss, grads = grad_fn(params, inputs)
        variables = variables.apply_gradients(grads=grads)
        return loss, variables

    variable_shardings = jax.tree_map(lambda x: AUTO(mesh), variable_avals)

    with autoshard(True):
        result_shape = jax.eval_shape(step_fn, variable_avals, batch_avals)
    out_shardings = jax.tree_map(lambda x: AUTO(mesh), result_shape)

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

    if local_batch is not None:

        def make_sharded_array(host_array, sharding):
            device_buffers = _put_to_devices(host_array, mesh.local_devices)
            shape = (
                host_array.shape[0] * num_data_loaders,
            ) + host_array.shape[1:]
            return jax.make_array_from_single_device_arrays(
                shape, sharding, device_buffers
            )

        def prepare_batch(local_batch_arrays):
            return jax.tree_map(
                make_sharded_array, local_batch_arrays, batch_shardings
            )

        initial_batch = prepare_batch(local_batch)
        with mesh, autoshard(True):
            init_fn = pjit(init_fn, out_shardings=variable_shardings)
            init_variables = init_fn(random.key(42), initial_batch)

        return compiled_step, init_variables, mesh, prepare_batch

    with mesh, autoshard(True):
        init_fn = pjit(init_fn, out_shardings=variable_shardings)
        init_variables = init_fn(random.key(42), global_batch)
    return compiled_step, init_variables, mesh
