# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import contextlib
import functools
import json
import re
from dataclasses import dataclass
from functools import partial
from typing import (
    Any,
    Callable,
    List,
    Literal,
    Mapping,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Type,
)

import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core
from jax._src import config as jax_config
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.mesh import thread_resources
from jax.experimental import shard_map as jax_shard_map
from jax.experimental.pjit import AUTO
from jax.interpreters import ad, mlir
from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P
from jax.tree import map as tree_map
from numpy.typing import NDArray

from .lib import (
    autoshard,
    optional_kwargs,
    should_ignore_transforms,
    with_sharding_constraint,
)
from .multimesh_jax_impl import (
    pop_task_context,
    push_task_context,
    register_task_factory,
)
from .no_op import no_op

jax_custom_partitioning = jax._src.custom_partitioning.custom_partitioning
jax_shard_map_shard_map = jax_shard_map.shard_map

_context_stack = []


class TaskMesh:
    """TaskMesh representing a submesh within the global mesh

    TaskMesh define the submesh devices, axis names, and axis sizes
    for a task that runs on a subset or slice of the global mesh

    Args:
        devices (Sequence[int]): The list of device numbers to include in
            the task. Device numbers correspond to a [O,N) relative
            numbering of devices within the executable, not global device
            numbers.
        axis_sizes (Optional[Sequence[int]]): The sizes of each
            task mesh axis. Defaults to None.
        axis_names (Optional[Sequence[str]]): The names of each
            task mesh axis. Defaults to None.
        abstract (Optional[AbstractMesh]): A Jax AbstractMesh
            defining both axis_sizes and axis_names. Defaults to None.
    """

    def __init__(
        self,
        *,
        devices: Sequence[int],
        axis_sizes: Optional[Sequence[int]] = None,
        axis_names: Optional[Sequence[str]] = None,
        abstract: Optional[AbstractMesh] = None,
    ):
        if abstract is None:
            self.mesh = AbstractMesh(axis_sizes, axis_names)
        else:
            self.mesh = abstract
        if not isinstance(devices, np.ndarray):
            devices = np.array(devices, dtype=int).reshape(
                self.mesh.axis_sizes
            )
        self.devices = devices

    def place(self, devices: Sequence[int]) -> Mesh:
        """Creates an equivalent task mesh placed on new devices

        Args:
            devices (Sequence[int]): The new devices to use for the mesh

        Returns:
            TaskMesh: A task mesh with the same axes placed on new devices
        """
        return TaskMesh(devices=devices, abstract=self.mesh)

    @property
    def abstract_mesh(self) -> AbstractMesh:
        return self.mesh


@dataclass
class Task:
    """Task dataclass defining all options for a named task context"""

    name: Optional[str] = None
    """The name of the task. If not given, a default task name based on the
       matched metadata name will be filled in."""
    mesh: Optional[TaskMesh] = None
    """The `TaskMesh` defining the submesh shape and axis names"""
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None
    """The mapping of logical axis names in the sharding spec
       to device axes in the task mesh given as priority-ordered pairs"""
    mesh_slice: Optional[Mapping[str, int]] = None
    """A mapping of axis: slice pairs defining the task submesh as
       a slice of the global mesh along the named axes"""
    extra_axes: Optional[Sequence[Tuple[str, str]]] = None
    """Extra logical->device axis mappings to use in addition to those
       defined by the mesh_slice"""
    split_backprop: Optional[Tuple[str, str]] = None
    """Whether to split backprop tasks into activation (critical path)
       and weight gradients (non-critical path)"""
    loop_dependent_mesh_slice: Optional[
        Callable[[int], Mapping[str, int]]
    ] = None
    """A callback that defines a different mesh slice for each
       iteration of the loop"""


class Context:
    def __init__(self):
        self._registered_task_slices: List[
            Tuple[Pattern, Callable[[str, bool], Task]]
        ] = []
        self.tasks = {}

    def __enter__(self):
        _context_stack.append(self)
        push_task_context()

        for regex, cpp_callback in self.tasks.items():
            register_task_factory(regex, cpp_callback)

    def __exit__(self, exc_type, exc_value, traceback):
        _context_stack.pop()
        pop_task_context()

    def find_matching_slice_in_scope(self) -> Optional[TaskMesh]:
        name_stack = (
            jax._src.interpreters.mlir._name_stack
            or jax._src.source_info_util.current().name_stack
        )

        scopes = []
        for x in name_stack.stack:
            x.wrap(scopes)

        full_scope = "/".join(scopes)
        for scope in reversed(scopes):
            for regex, task_callback in self._registered_task_slices:
                if match := regex.search(scope):
                    backprop = "transpose(jvp" in full_scope
                    task = task_callback(match.groups()[0], backprop)
                    if task.mesh_slice is None:
                        raise ValueError(
                            f"task {match.groups()[0]} does not "
                            f"return a mesh slice: {task}"
                        )
                    return task.mesh_slice
        return None

    def register_task(
        self,
        matcher: str,
        *,
        callback: Optional[Callable[[str, bool], Task]] = None,
        task: Optional[Task] = None,
    ):
        """Registers a name or regex-based task autosharding context

        Args:
            matcher: A name or regular expression with match group.
                This should match the name of a Flax module or a name passed to
                ``jax.with_named_scope``.
            callback: optional, a callback taking the match group from ``matcher``
                    and a bool indicating whether it is backprop. The
                    callback must return a ``Task`` defining
                    the mesh and other attributes of the task scope
            task: optional, a ``Task`` defining the mesh and other attributes
                    of the task scope

        Returns:
            None

        Examples:
            Basic usage with mesh argument is:

            >>> import numpy as np
            >>> import jax
            >>> from multimesh.jax import register_task
            >>> from jax.sharding import PartitionSpec as P, AbstractMesh, Mesh
            >>>
            >>> def f(x):
            ...   with jax.named_scope("subtask"):
            ...     x = with_sharding_constraint(x, P("batch", "model"))
            ...     out = x*x
            ...     return with_sharding_constraint(out, P("batch", "model"))
            >>>
            >>> def callback(name: str, backprop: bool):
            >>>    mesh = TaskMesh(devices=(0,1,2,3), axis_names=("x", "y"), axis_sizes=(2,2))
            >>>    return Task(mesh=mesh, name=name)
            >>> register_task("subtask", callback=callback)
        """  # noqa: E501

        if "(" not in matcher and ")" not in matcher:
            regex = f"({matcher})"
        else:
            regex = matcher

        if callback is None and task is None:
            raise Exception("must specify callback or task to register_task")
        else:

            def cpp_callback(matched: str, backprop: bool):
                mm_task = task or callback(matched, backprop)
                if not isinstance(mm_task, Task):
                    raise ValueError(
                        "MultiMesh task callback must return a Task object"
                    )

                if mm_task.mesh_slice is not None and mm_task.mesh is not None:
                    raise ValueError(
                        "cannot give both a task.mesh and task.mesh_slice"
                        f" for task matcher {matcher}"
                    )
                if mm_task.mesh_slice is not None:
                    task_mesh = slice_global_mesh(**mm_task.mesh_slice)
                elif mm_task.mesh is not None:
                    task_mesh = mm_task.mesh
                else:
                    raise ValueError(
                        "both task.mesh and task.mesh_slice are None"
                        f" for task matcher {matcher}"
                    )

                if mm_task.loop_dependent_mesh_slice is not None:

                    def loop_dependent_devices(i):
                        iter_mesh_slice = mm_task.loop_dependent_mesh_slice(i)
                        iter_submesh = slice_global_mesh(**iter_mesh_slice)
                        submesh_devs = iter_submesh.devices.flatten()

                        num_default_devs = task_mesh.devices.size
                        if submesh_devs.size != num_default_devs:
                            raise ValueError(
                                "loop dependent submesh"
                                f" ({submesh_devs.size} devices)"
                                " must have same number of devices as default"
                                f" task mesh ({num_default_devs} devices)"
                                f" for task matcher {matcher}"
                            )

                        return submesh_devs

                else:
                    loop_dependent_devices = None

                dims = list(task_mesh.mesh.shape.values())
                device_axes = task_mesh.mesh.axis_names
                logical_axes = mm_task.logical_axes or list(
                    (dim, dim) for dim in device_axes
                )
                if mm_task.extra_axes is not None:
                    for logical, device in mm_task.extra_axes:
                        logical_axes.append((logical, device))

                name = mm_task.name or matcher
                return (
                    name,
                    task_mesh.devices.flatten(),
                    dims,
                    device_axes,
                    logical_axes,
                    mm_task.split_backprop,
                    loop_dependent_devices,
                )

            if callback is None:

                def task_callback(x, y):
                    return task

            else:
                task_callback = callback
            self._registered_task_slices.append(
                (re.compile(regex), task_callback)
            )
        self.tasks[regex] = cpp_callback


class MultiMesh(Context, contextlib.ContextDecorator):
    """MultiMesh represents a global mesh with named submeshes

    MultiMesh defines a global mesh with named submeshes. The mesh can
    can be sliced along certain dimensions to create submeshes with a
    specific subshape. The MultiMesh can either create a new global
    mesh or wrap an existing jax.sharding.Mesh.

    Args:
        axis_sizes (Optional[Sequence[int]]): The sizes of each
            global mesh axis. Defaults to None. Ignored if `global_mesh`
            is given.
        axis_names (Optional[Sequence[str]]): The names of each
            global mesh axis. Defaults to None. Ignored if `global_mesh`
            is given.
        devices (Optional[np.array[xc.Device]]): The shaped array of devices
            to include in the global mesh. The shape should match
            `axis_sizes`. Defaults to None. If not specified,
            the default `jax.devices()` will be used and reshaped to
            match the `axis_sizes`. Ignored if `global_mesh` is given.
        global_mesh (Optional[Mesh]): A Jax Mesh
            defining devices, axis_sizes, and axis_names. Defaults to None.
            If given, all other parameters will be ignored. If not
            given, then `axis_sizes` and `axis_names` must be given.
    """

    def __init__(
        self,
        *,
        axis_sizes: Optional[Sequence[int]] = None,
        axis_names: Optional[Sequence[str]] = None,
        devices: Optional[NDArray[xc.Device]] = None,
        global_mesh: Optional[Mesh] = None,
    ):
        super().__init__()
        if global_mesh:
            self.global_mesh = global_mesh
        else:
            if devices is None:
                devices = np.array(jax.devices()).reshape(axis_sizes)
            self.global_mesh = Mesh(devices, axis_names)

    def slice(self, **kwargs: Mapping[str, int]) -> Mesh:
        """Slice the global mesh along the specified axis.

        Slices a mesh along the given named dimensions as
        defined by the kwargs map.

        Args:
            kwargs: (Mapping[str,int]). A mapping defining
            axis: value pairs that will be sliced out of
            the global mesh.

        Returns:
            Mesh: A submesh sliced along the specified dimensions.
        """
        slices = tuple(
            slice(None)
            if ax not in kwargs
            else slice(kwargs[ax], kwargs[ax] + 1)
            for ax in self.axis_names
        )
        devices = self.global_mesh.devices[slices]
        return Mesh(devices, self.global_mesh.axis_names)

    def single_slice_ids(self, **kwargs) -> List[int]:
        return [
            dev.id for dev in self.single_slice_devices(**kwargs).flatten()
        ]

    def single_slice_devices(self, **kwargs) -> np.ndarray:
        slices = tuple(
            slice(None)
            if ax not in kwargs
            else slice(kwargs[ax], kwargs[ax] + 1)
            for ax in self.axis_names
        )
        return self.global_mesh.devices[slices]

    @property
    def shape_tuple(self):
        return tuple(
            (name, size)
            for name, size in zip(self.axis_names, self.devices.shape)
        )

    @property
    def size(self):
        return self.global_mesh.size

    @property
    def axis_types(self):
        return self.global_mesh.axis_types

    @property
    def _are_all_axes_manual(self) -> bool:
        return self.global_mesh._are_all_axes_manual

    @property
    def _are_all_axes_auto(self) -> bool:
        return self.global_mesh._are_all_axes_auto

    @property
    def _are_all_axes_explicit(self) -> bool:
        return self.global_mesh._are_all_axes_explicit

    @property
    def _are_all_axes_auto_or_manual(self) -> bool:
        return self.global_mesh._are_all_axes_auto_or_manual

    @property
    def _any_axis_manual(self) -> bool:
        return self.global_mesh._any_axis_manual

    @property
    def _any_axis_auto(self) -> bool:
        return self.global_mesh._any_axis_auto

    @property
    def _any_axis_explicit(self) -> bool:
        return self.global_mesh._any_axis_explicit

    @property
    def _any_axis_auto_or_manual(self) -> bool:
        return self.global_mesh._any_axis_auto_or_manual

    @property
    def auto_axes(self):
        return self.global_mesh.auto_axes

    @property
    def explicit_axes(self):
        return self.global_mesh.explicit_axes

    @property
    def manual_axes(self):
        return self.global_mesh.manual_axes

    @property
    def _axis_types_dict(self):
        return self.global_mesh._axis_types_dict

    @functools.cached_property
    def abstract_mesh(self):
        return AbstractMesh(
            self.axis_sizes, self.axis_names, axis_types=self.axis_types
        )

    @property
    def is_multi_process(self):
        return self.global_mesh.is_multi_process

    @property
    def _flat_devices_tuple(self):
        return self.global_mesh._flat_devices_tuple

    @property
    def _flat_devices_set(self):
        return self.global_mesh._flat_devices_set

    @property
    def local_devices(self):
        return self.global_mesh.local_devices

    @property
    def axis_names(self):
        return self.global_mesh.axis_names

    @functools.cached_property
    def _name_to_type(self):
        return self.global_mesh._name_to_type

    @property
    def axis_sizes(self) -> tuple[int, ...]:
        return self.global_mesh.axis_sizes

    @property
    def devices(self):
        return self.global_mesh.devices

    @property
    def shape(self):
        return self.global_mesh.shape

    @property
    def _internal_device_list(self):
        return self.global_mesh._internal_device_list

    @property
    def empty(self):
        return self.global_mesh.empty

    def __enter__(self):
        self.autoshard = autoshard(True)
        self.autoshard.__enter__()
        Context.__enter__(self)
        new_env = thread_resources.stack[-1].with_mesh(self)
        thread_resources.stack.append(new_env)
        thread_resources.env = new_env
        jax_config.mesh_context_manager.set_local(
            tuple(
                t.physical_mesh
                for t in thread_resources.stack
                if not t.physical_mesh.empty
            )
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        Context.__exit__(self, exc_type, exc_value, traceback)
        self.autoshard.__exit__(exc_type, exc_value, traceback)
        thread_resources.stack.pop()
        thread_resources.env = thread_resources.stack[-1]
        jax_config.mesh_context_manager.set_local(
            tuple(
                t.physical_mesh
                for t in thread_resources.stack
                if not t.physical_mesh.empty
            )
        )
        return False

    def __repr__(self):
        return f"MultiMesh({repr(self.global_mesh)})"

    def __str__(self):
        return f"MultiMesh({str(self.global_mesh)})"


class custom_partitioning:
    def __init__(self, *args, **kwargs):
        self.partitioner = jax_custom_partitioning(*args, **kwargs)

    def def_partition(self, *args, **kwargs):
        return self.partitioner.def_partition(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        mesh = jax._src.mesh.thread_resources.env.physical_mesh
        if not isinstance(mesh, MultiMesh):
            raise Exception("The global mesh context should a MultiMesh")

        # see if any of the contexts define a matching scope
        # starting with the innermost scope and working outwards
        for context in reversed(_context_stack):
            slices = context.find_matching_slice_in_scope()
            if slices:
                with mesh.slice(**slices):
                    return self.partitioner(
                        *args,
                        **kwargs,
                    )
        # no slice context found
        with mesh:
            return self.partitioner(*args, **kwargs)


def shard_map(*args, mesh=None, **kwargs):
    mesh = mesh or jax._src.mesh.thread_resources.env.physical_mesh
    for context in reversed(_context_stack):
        slices = context.find_matching_slice_in_scope() or {}
        if slices:
            return jax_shard_map_shard_map(
                *args, mesh=mesh.slice(**slices), **kwargs
            )
    return jax_shard_map_shard_map(*args, mesh=mesh)


def parallelize(
    fun: Callable,
    *,
    init_params: Optional[Callable[[], Any]] = None,
    get_input_batch: Optional[Callable[[], Any]] = None,
    initial_batch: Optional[Any] = None,
    devices: Optional[Sequence[xc.Device] | np.ndarray] = None,
):
    """Compiles an abstract function into a sharded function

    Parallelizes a function with logical sharding annotations into
    a function with explicit device shardings for all inputs
    and parameters. Input and output shapes are derived from the
    ``init_params`` and ``get_input_batch`` functions. Functions
    to be parallelized must conform to a standard format in which
    parameters are the first argument and input batches are the
    seocnd argument. This matches the format for the ``grad``
    transformation which differentiates the first argument
    and assumes the second (and later) arguments are input
    batches.

    Args:
      fun: Function to be parallelized. ``fun`` should be pure.
        See documentation for `jax.jit`_ for requirements for ``fun``.
      init_params: optional, a function taking no arguments that generates
       input parameters without shardings. Optional if ``fun`` does not take
       parameters as a first argument.
      get_input_batch: optional, a function taking no arguments that generates
        input batches without shardings. Optional if ``fun`` does not take
        input batches as a second argument or if ``initial_batch`` is given
        instead.
      initial_batch: optional, an array or pytree of arrays with shardings
        valid as input parameters to ``fun``
      devices: optional, a numpy array or list of jax devices
        specifying the devices to parallelize over. If not given,
        the function is parallelized over all devices.

    Returns:
      A tuple of (sharded_fun, sharded_init_params, sharded_get_input_batch)
      if ``get_input_batch`` is given or a tuple
      (sharded_fun, sharded_init_params) if a sharded
      ``initial_batch`` is given.

    Examples:
      >>> import jax.numpy as jnp
      >>> import jax
      >>> from multimesh.jax import parallelize, task, with_sharding_constraint
      >>> from jax.sharding import PartitionSpec as P
      >>>
      >>> def f(x,y):
      ...   x = with_sharding_constraint(x, P("x",))
      ...   y = with_sharding_constraint(y, P("x",))
      ...   out = x*y
      ...   return with_sharding_constraint(out, P("batch", "model"))
      >>>
      >>> devices = np.array(jax.devices()).reshape(2,2)
      >>> task_f = task(f, devices=devices, device_axes=("x",))
      >>>
      >>> def init():
      >>>   return jnp.arange(16)
      >>>
      >>> sh_f, sh_init_params, sh_init_batch = parallelize(f, init, init)

    .. _jax.jit: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
    """  # noqa: E501
    if devices is None:
        devices = jax.devices()

    flat_devices = np.asarray(devices).flatten()
    mesh = Mesh(flat_devices, ["x"])

    with autoshard(True):
        param_shapes = jax.eval_shape(init_params)
        abstract_params = tree_map(
            lambda x: jax.core.ShapedArray(x.shape, x.dtype), param_shapes
        )

        param_shardings = tree_map(lambda x: AUTO(mesh), param_shapes)
        if initial_batch is None:
            if get_input_batch is None:
                raise ValueError(
                    "multimesh.jax.parallelize requires either get_input_batch"
                    "function or initial_batch parameter"
                )

            batch_shapes = jax.eval_shape(get_input_batch)
            batch_sharding = tree_map(lambda x: AUTO(mesh), batch_shapes)
            initial_batch = tree_map(
                lambda x: jax.core.ShapedArray(x.shape, x.dtype), batch_shapes
            )
        else:
            batch_sharding = tree_map(lambda x: x.sharding, initial_batch)
            batch_shapes = tree_map(
                lambda x: jax.core.ShapedArray(x.shape, x.dtype), initial_batch
            )

        result_shape = jax.eval_shape(fun, param_shapes, batch_shapes)
        out_shardings = tree_map(lambda x: AUTO(mesh), result_shape)

        jit_f = jax.jit(
            fun,
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


def __mm_task_lowering_impl(*args, jaxpr, **unused_kwargs):
    del unused_kwargs
    return jax_core.jaxpr_as_fun(jaxpr)(*args)


def _custom_abstract_eval(*args, jaxpr, **unused_kwargs):
    del unused_kwargs
    del args
    return jaxpr.out_avals


mm_task_p = jax.extend.core.Primitive("mm_task")
mm_task_p.multiple_results = True
mm_task_p.def_abstract_eval(_custom_abstract_eval)
mm_task_p.def_impl(__mm_task_lowering_impl)


def call_mm_task(
    f, *args, config: str = "", name_override: Optional[str] = None, **kwargs
):
    jaxpr, out_shapes = jax.make_jaxpr(
        partial(f, **kwargs), return_shape=True
    )(*args)
    flat_args = jax.tree.leaves(args)
    out_tree = jax.tree.structure(out_shapes)
    out_flat = mm_task_p.bind(
        *flat_args,
        name=name_override or f.__name__,
        jaxpr=jaxpr,
        config=config,
    )
    return jax.tree.unflatten(out_tree, out_flat)


def call_mm_task_fwd(
    f, *args, config: str = "", name_override: Optional[str] = None, **kwargs
):
    return (
        call_mm_task(
            f, *args, config=config, name_override=name_override, **kwargs
        ),
        args,
    )


def call_mm_task_bwd(
    f,
    primals,
    tangents,
    config: str = "",
    name_override: Optional[str] = None,
    **kwargs,
):
    return call_mm_task(
        f,
        primals,
        tangents,
        config=config,
        name_override=name_override,
        **kwargs,
    )


def _mm_task_lowering(
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
    mm_task = hlo.CustomCallOp(
        [r.type for r in call_op.results],
        call_op.operands,
        call_target_name="MultiMeshTask",
        called_computations=ir.ArrayAttr.get([called_fn]),
        backend_config=ir.StringAttr.get(config),
    )
    return mm_task.results


mlir.register_lowering(mm_task_p, _mm_task_lowering)


def mm_task_linear(ct, _, **kwargs):
    return (mm_task_p.bind(ct, **kwargs),)


ad.deflinear2(mm_task_p, mm_task_linear)


def _get_wrapped_task(
    fxn,
    name: str,
    *,
    devices: Sequence[int],
    axis_sizes: Sequence[int],
    axis_names: Sequence[str],
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None,
):
    args = dict(
        name=name,
        devices=devices,
        autosharding=dict(
            dims=axis_sizes,
            device_axes=axis_names,
            logical_axes=logical_axes,
        ),
    )

    wrapped = partial(call_mm_task, fxn, config=json.dumps(args))

    fwd = partial(call_mm_task_fwd, fxn, config=json.dumps(args))

    def f_bwd(primals, tangents):
        _, f_vjp = jax.vjp(fxn, *primals)
        return f_vjp(tangents)

    bwd = partial(
        call_mm_task_bwd,
        f_bwd,
        config=json.dumps(args),
        name_override=f"{fxn.__name__}_bwd",
    )

    vjp_taskify = jax.custom_vjp(wrapped)
    vjp_taskify.defvjp(fwd, bwd)
    return vjp_taskify


def task(
    fun: Callable | Type,
    name: Optional[str] = None,
    *,
    task: Optional[Task] = None,
    **kwargs,
):
    """Wraps a function in an auto-sharding task context

    Args:
      fun: Function to be encapsulated as a task. ``fun`` should be pure.
        See documentation for `jax.jit`_ for requirements for ``fun``.
      name: optional, a metadata name to assign to the task context
      task: the `Task` dataclass defining the submesh and sharding
      kwargs: options passed through to `Task` dataclass constructor

    Returns:
      A wrapped version of ``fun`` usable as a submesh task.

    Examples:
      >>> import numpy as np
      >>> import jax
      >>> from multimesh.jax import task, with_sharding_constraint
      >>> from jax.sharding import PartitionSpec as P
      >>>
      >>> def f(x):
      ...   x = with_sharding_constraint(x, P("batch", "model"))
      ...   out = x*x
      ...   return with_sharding_constraint(out, P("batch", "model"))
      >>>
      >>> devices = np.array(jax.devices()).reshape(2,2)
      >>> task_f = task(f, devices=devices,
      ...               device_axes=("x", "y"),
      ...               logical_axes=(
      ...                 ("batch", "x"),
      ...                 ("model", "y"),
      ...               ))
      >>>

    .. _jax.jit: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
    """  # noqa: E501
    if should_ignore_transforms():
        return fun

    if name is None:
        name = fun.__name__

    if task is None:
        task = Task(**kwargs)

    if task.mesh is None:
        devices = tuple(range(0, jax.device_count()))
        axis_names = ("x",)
        axis_sizes = (len(devices),)
        logical_axes = (("x", "x"),)
    else:
        devices = tuple(x.item() for x in task.mesh.devices.flatten())
        axis_names = task.mesh.mesh.axis_names
        axis_sizes = task.mesh.mesh.axis_sizes
        if task.logical_axes is None:
            logical_axes = ((ax, ax) for ax in axis_names)
        else:
            logical_axes = tuple(task.logical_axes)

        if task.extra_axes is not None:
            logical_axes = logical_axes + tuple(task.extra_axes)

    return _get_wrapped_task(
        fun,
        name,
        devices=devices,
        axis_sizes=axis_sizes,
        axis_names=axis_names,
        logical_axes=logical_axes,
    )


def microbatch(
    fun,
    dim: int,
    size: int,
    argnum: int = 0,
    interleave: Optional[int] = None,
    num_stages: Optional[int] = None,
    schedule: Optional[
        Literal["1f1b", "gpipe", "wavefront", "custom", "zero-bubble-h2"]
    ] = None,
    unrolling: Optional[int] = None,
    arg_shardings: Optional[Any] = None,
    custom_schedule: Optional[
        list[list[str]] | list[list[tuple[int, str]]]
    ] = None,
):
    """Unrolls a function along an axis into a microbatch loop.

    The input tensors are sliced along the axis for each microbatch.
    The results of each microbatch are sum-reduced to produce the final
    result. The output tensors should be equivalent (modulo precision)
    to the function without the transform. No semantic checking is
    currently done on the function to ensure that sum-reduction of
    microbatches is equivalent to the original function and relies
    on the user ensuring the transformation is equivalent.

    For a single input/output, the transformation is equivalent to:

    .. code-block:: python

      import jax
      import jax.numpy as jnp
      def microbatch_f(x, params):
        result = jax.eval_shape(fun, x, params)
        accumulator = jnp.zeros(result.shape, dtype=result.dtype)
        num_loops = x.shape[dim] // size
        for i in range(num_loops):
            microbatch_x = jnp.dynamic_slice(x, ...)
            accumulator += f(microbatch_x, params)
        return accumulator

    A microbatch loop will usually be a nested loop of N iterations
    over S stages:

    .. code-block:: python

      for mb in range(num_microbatches):
        for stage in range(num_stages):
            ...

    Microbatches are assumed to be independent and the iteration
    order for ``mb`` is arbitary. Microbatches can be tiled
    or unrolled in any order with the stages, e.g.

    .. code-block:: python

      for block in range(blocks):
        for stage in range(num_stages):
          for mb in range(unrolling):
            ...

    The structure of the nested loops can be tuned
    by specifying ``schedule``, ``interleave``, ``unrolling``,
    and ``num_stages`` parameters. If left unspecified, the
    compiler/runtime is free to choose the microbatch schedule.

    Args:
      fun: Function to be transformed into microbatch loops.
        ``fun`` should be pure.
        See documentation for `jax.jit`_ for requirements for ``fun``.
      dim: an int specifying which dimension of the input tensor(s) should
           be sliced for each microbatch
      size: an int specifying the size of the microbatch dimension
            for each microbatch
      argnum: optional, the argument number that will be sliced for each
        microbatch. If the argument is a pytree of tensors rather than a single
        tensor, then all tensors in the tree are sliced along the given ``dim``.
        All other arguments to ``fun`` are unmodified.
      interleave: optional, a hint to the microbatch scheduler about how
        tasks within the loop should be interleavd.
      schedule: optional, a string identifying the schedule of microbatch
        iterations/stages such as 'gpipe' or '1f1b'.
      unrolling: optional, an int specifying the unrolling of the microbatch
        loop. By default, the microbatch loop is fully unrolled. Only
        relevant for the `gpipe` schedule.
      arg_shardings: optional, an object or (prefix) pytree of objects matching
        ``argnum`` with shardings. The shardings can be any sharding-equivalent
        object including partition specs or ``NamedSharding``  If specified,
        this applies the sharding annotations to all sliced inputs.
      custom_schedule: optional, a list of lists of strings or tuples of
        (int, string) specifying the custom pipeline schedule of microbatch
        iterations/stages when ``schedule`` is 'custom'. The custom schedule
        Each list in custom_schedule corresponds to a device mesh, and each
        element in that list is a task name, optionally associated with a
        specific microbatch. There must be exactly `pipeline_depth` device
        meshes in the custom schedule, with each task appearing exactly
        `num_microbatches` times in the schedule.

    Returns:
      A wrapped version of ``fun`` that executes as a microbatch loop.

    .. _jax.jit: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
    """  # noqa: E501

    if custom_schedule is not None:
        if schedule is not None and schedule != "custom":
            raise ValueError(
                f"When custom_schedule is provided, schedule must be "
                f"'custom' or None, but got '{schedule}'"
            )
        # Set schedule to "custom" by default when custom_schedule is provided
        schedule = "custom"

    if should_ignore_transforms():
        return fun

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
            return fun(*args, **kwargs)

        def is_list_of_list_of_strings(v: list[Any]) -> bool:
            return isinstance(v, list) and all(
                isinstance(i, list) and all(isinstance(j, str) for j in i)
                for i in v
            )

        nonlocal custom_schedule
        if custom_schedule is not None and is_list_of_list_of_strings(
            custom_schedule
        ):
            canonical_custom_schedule = []
            for device_row in custom_schedule:
                task_counter = collections.defaultdict(int)
                canonical_custom_schedule.append([])
                for task_id in device_row:
                    canonical_custom_schedule[-1].append(
                        (task_counter[task_id], task_id)
                    )
                    task_counter[task_id] += 1
                for task_id, count in task_counter.items():
                    if count != num_microbatches:
                        raise ValueError(
                            f"task {task_id} has {count} microbatches, "
                            f"but there are {num_microbatches} microbatches"
                        )
            custom_schedule = canonical_custom_schedule

        json_args = optional_kwargs(
            num_microbatches=num_microbatches,
            slice_dim=dim,
            size=size,
            batch_dim=dim,
            interleave=interleave,
            unrolling=unrolling,
            num_stages=num_stages,
            schedule=schedule,
            custom_schedule=custom_schedule,
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
            pre_slice_shardings = tree_map(lambda a: arg_shardings, x)
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
        result_shapes = jax.eval_shape(fun, *new_args, **kwargs)

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
            results = fun(*new_args, **kwargs)
            flat_results, _ = jax.tree_util.tree_flatten(results)

            new_results = [x + y for x, y in zip(flat_prev, flat_results)]
            return (offset, jax.tree_util.tree_unflatten(treedef, new_results))

        offset = 0
        offset, result = jax.lax.fori_loop(
            0, num_microbatches, body_fun, (offset, initial_results)
        )

        return result

    return wrapped


def get_global_mesh() -> Mesh:
    return jax._src.mesh.thread_resources.env.physical_mesh


def get_abstract_global_mesh():
    return get_global_mesh.abstract_mesh()


def slice_global_mesh(**to_slice) -> TaskMesh:
    mesh = get_global_mesh()
    shape = list(mesh.shape.values())
    all_devices = np.arange(mesh.size).reshape(shape)
    slices = tuple(
        slice(to_slice[name], to_slice[name] + 1)
        if name in to_slice
        else slice(None)
        for name in mesh.axis_names
    )
    devices = all_devices[slices]
    return TaskMesh(
        axis_names=mesh.axis_names, axis_sizes=devices.shape, devices=devices
    )


# always have a default context on the stack
_context_stack.append(Context())


def register_task(
    matcher: str,
    *,
    callback: Optional[Callable[[str, bool], Task]] = None,
    task: Optional[Task] = None,
):
    _context_stack[-1].register_task(matcher, callback=callback, task=task)
    if len(_context_stack) == 1:
        register_task_factory(matcher, _context_stack[-1].tasks[matcher])


register_task.__doc__ = _context_stack[-1].register_task.__doc__
