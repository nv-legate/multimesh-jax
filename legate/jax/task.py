import json
from functools import partial
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Type

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

from .legate_jax_impl import (
    _register_metadata_name,
    _register_task,
    _register_task_factory,
)
from .lib import (
    autoshard,
    optional_kwargs,
    should_ignore_transforms,
    with_sharding_constraint,
)
from .no_op import no_op


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
      >>> from legate.jax import parallelize, task, with_sharding_constraint
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

        result_shape = jax.eval_shape(fun, param_shapes, batch_shapes)
        out_shardings = jax.tree_map(lambda x: AUTO(mesh), result_shape)

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

    if logical_axes is not None and device_axes is None:
        raise ValueError(f"task {name} given logical_axes but no device_axes")

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
    fun: Callable | Type,
    name: Optional[str] = None,
    *,
    mesh: Optional[Mesh] = None,
    devices: np.ndarray | Sequence[xc.Device] | None = None,
    device_axes: Optional[Sequence[str]] = None,
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None,
):
    """Wraps a function in an auto-sharding task context

    Args:
      fun: Function to be encapsulated as a task. ``fun`` should be pure.
        See documentation for `jax.jit`_ for requirements for ``fun``.
      name: optional, a metadata name to assign to the task context
      mesh: optional, a Mesh context defining the devices and mesh shape
      devices: optional, a numpy array or list of jax devices
        specifying the devices to include in the task submesh.
        One of ``mesh`` or ``devices`` must be given. If ``devices``
        is a numpy array, the mesh shape is inferred from the shape
        of the device array.
      device_axes: optional, a list of names to assign to each device axis.
        The number of names must match the shape of ``mesh`` or ``devices``.
        If this and ``mesh`` are not given, logical sharding constraints
        will be translated to replicated sharding.
      logical_axes: optional, a list of string pairs ('logical', 'device')
        giving the translation from logical names to physical device names.
        The logical names should match those passed to
        ``with_sharding_constraint`` calls within the task. If None,
        the device axis names are used directly for autosharding.
        Raises a ``ValueError`` if ``logical_axes`` are given
        but no ``mesh`` or ``device_axes`` are specified.

    Returns:
      A wrapped version of ``fun`` usable as a submesh task.

    Examples:
      >>> import numpy as np
      >>> import jax
      >>> from legate.jax import task, with_sharding_constraint
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

    return _get_wrapped_task(
        fun,
        name,
        mesh=mesh,
        devices=devices,
        device_axes=device_axes,
        logical_axes=logical_axes,
    )


def microbatch(
    fun,
    dim: int,
    size: int,
    argnum: int = 0,
    interleave: Optional[int] = None,
    num_stages: Optional[int] = None,
    schedule: Optional[Literal["1f1b", "gpipe", "wavefront"]] = None,
    unrolling: Optional[int] = None,
    arg_shardings: Optional[Any] = None,
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

    Returns:
      A wrapped version of ``fun`` that executes as a microbatch loop.

    .. _jax.jit: https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html
    """  # noqa: E501
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


def register_task(
    regex: str,
    *,
    name: Optional[str] = None,
    mesh: Optional[Mesh] = None,
    dims: Optional[Sequence[int]] = None,
    callback: Optional[Callable[[str], list[int]]] = None,
    devices: np.ndarray | Sequence[xc.Device] | Sequence[int] | None = None,
    device_axes: Optional[Sequence[str]] = None,
    logical_axes: Optional[Sequence[Tuple[str, str]]] = None,
    fusion_color: int = 0,
    loop_submesh_size: Optional[int] = None,
    loop_submesh_reverse: bool = False,
):
    """Registers a name or regex-based task autosharding context

    Args:
      regex: A full name or regular expression with match group.
        This should match the name of a Flax module or a name passed to
        ``jax.with_named_scope``.
      mesh: optional, a Mesh context defining the devices and mesh shape
      callback: optional, a function taking the match group from
        the ``regex`` and returning an integer list
        enumerating the devices to include in the task.
      dims: optional, the submesh dimensions for the task. Only one
        of ``mesh`` or ``dims`` should be given.
      devices: optional, a numpy array or list of jax devices
        specifying the devices to include in the task submesh.
        One of ``mesh`` or ``devices`` or ``callback`` must be given.
        If ``devices`` is a numpy array, the mesh shape is inferred from the shape
        of the device array.  If both ``mesh`` and ``devices`` are given,
        then ``mesh`` is considered to define a submesh of the ``devices``
        for task instances within a loop. The function will then infer
        a ``loop_submesh_size``.  Similarly, ``dims`` can specify
        a submesh smaller than ``devices``.
      device_axes: optional, a list of names to assign to each device axis.
        The number of names must match the shape of ``devices``.
        User must give only one of ``mesh`` or ``device_axes``.
        If this and ``mesh`` are not given, logical sharding constraints
        will be translated to replicated sharding.
      logical_axes: optional, a list of string pairs ('logical', 'device')
        giving the translation from logical names to physical device names.
        The logical names should match those passed to
        ``with_sharding_constraint`` calls within the task. If None,
        the device axis names are used directly for autosharding.
        Raises a ``ValueError`` if ``logical_axes`` are given
        but no ``device_axes`` or ``mesh`` are specified.
      fusion_color: optional, an integer specifying which tasks can
        be fused together by the compiler. Only tasks with the same
        ``fusion_color`` can be fused.
      loop_submesh_size: optional, an integer specifying that each instance
        of this task inside a loop should be rotate amongst submeshes
        that are smaller than ``mesh`` or ``devices``. If ``loop_sumbesh_size``
        is 2 and ``devices`` is [0,1,2,3], then tasks will rotate
        between submeshes [0,1] and [2,3].
      loop_submesh_reverse: optional, whether submesh devices should be rotated
        by counting 0...N or reversed to rotate N...0.

    Returns: None

    Examples:
      Basic usage with mesh argument is:

      >>> import numpy as np
      >>> import jax
      >>> from legate.jax import register_task
      >>> from jax.sharding import PartitionSpec as P, Mesh
      >>>
      >>> def f(x):
      ...   with jax.named_scope("subtask"):
      ...     x = with_sharding_constraint(x, P("batch", "model"))
      ...     out = x*x
      ...     return with_sharding_constraint(out, P("batch", "model"))
      >>>
      >>> devices = np.array(jax.devices()).reshape(2,2)
      >>> mesh = Mesh(devices, ("x", "y"))
      >>> register_task("subtask",
      ...               mesh=mesh,
      ...               logical_axes=(
      ...                 ("batch", "x"),
      ...                 ("model", "y"),
      ...               ))

      A callback function can be used to dynamically compute devices:

      >>> import numpy as np
      >>> import jax
      >>> from legate.jax import register_task
      >>> from jax.sharding import PartitionSpec as P, Mesh
      >>>
      >>> def callback(name):
      ...   layer_num = int(name.split(".")][-1])
      ...   devices_per_layer = 4
      ...   start = layer_num * devices_per_layer
      ...   return list(range(start, start + devices_per_layer))
      >>>
      >>> def f(x):
      ...   x = with_sharding_constraint(x, P("batch", "model"))
      ...   out = x*x
      ...   return with_sharding_constraint(out, P("batch", "model"))
      >>>
      >>> def c(x):
      ...   with jax.named_scope("layer.0"):
      ...     x = f(x)
      ...   with jax.named_scope("layer.1"):
      ...     return f(x)
      >>>
      >>> register_task("subtask",
      ...               callback=callback,
      ...               dims=(2,2),
      ...               device_axes=("x","y"),
      ...               logical_axes=(
      ...                 ("batch", "x"),
      ...                 ("model", "y"),
      ...               ))

      A submesh and global device list can be given to indicate that
      multiple instances of the task within a loop should be rotated
      to different submeshes

      >>> import numpy as np
      >>> import jax
      >>> from legate.jax import register_task
      >>> from jax.sharding import PartitionSpec as P, Mesh
      >>>
      >>> def f(x):
      ...   with jax.named_scope("subtask"):
      ...     x = with_sharding_constraint(x, P("batch", "model"))
      ...     out = x*x
      ...     return with_sharding_constraint(out, P("batch", "model"))
      >>>
      >>> devices = np.array(jax.devices())
      >>> mesh = Mesh(devices[0:2].reshape(2,1), ("x", "y"))
      >>> register_task("subtask",
      ...               mesh=mesh,
      ...               devices=devices,
      ...               logical_axes=(
      ...                 ("batch", "x"),
      ...                 ("model", "y"),
      ...               ))
    """  # noqa: E501

    def _to_device_id(d: int | xc.Device):
        if isinstance(d, int):
            return d
        if isinstance(d, xc.Device):
            return d.id
        raise ValueError(
            f"in register_task({regex}), {d} is not a device ID or xc.Device"
        )

    if device_axes is not None and mesh is not None:
        raise ValueError(
            f"in register_task({regex}), cannot give both device_axes and mesh"
        )

    if mesh is not None:
        device_ids = [d.id for d in mesh.devices.flatten()]
        dims = mesh.devices.shape
        device_axes = mesh.axis_names
        if devices is not None:
            loop_submesh_size = len(device_ids)
            if isinstance(devices, np.ndarray):
                device_ids = [_to_device_id(d) for d in devices.flatten()]
            else:
                device_ids = [_to_device_id(d) for d in devices]
            loop_submesh_reverse = devices[0] != mesh.devices[0]
    elif devices is not None:
        if isinstance(devices, np.ndarray):
            device_ids = [_to_device_id(d) for d in devices.flatten()]
            if dims is None:
                dims = devices.shape
            dim_prod = np.prod(dims)
            if dim_prod < len(devices):
                loop_submesh_size = dim_prod
        else:
            if dims is None:
                dims = (len(devices),)
            device_ids = [_to_device_id(d) for d in devices]
    elif callback is not None:
        if dims is None:
            raise ValueError(
                f"in register_task({regex}), when callback is used, you must "
                "specify mesh, devices, or dims to define mesh shape"
            )
    else:
        raise ValueError(
            f"in register_task({regex}), must give mesh, devices, or callback"
        )

    if device_axes is None and mesh is None:
        raise ValueError(
            f"in register_task({regex}), must give either device_axes and mesh"
        )

    if logical_axes is None:
        logical_axes = [(ax, ax) for ax in device_axes]

    _register_metadata_name(regex, name)
    if callback is not None:
        _register_task_factory(
            name or regex,
            callback,
            list(dims),
            list(device_axes),
            list(logical_axes),
            fusion_color,
        )
    else:
        _register_task(
            name or regex,
            device_ids,
            list(dims),
            list(device_axes),
            list(logical_axes),
            fusion_color,
            loop_submesh_size,
            loop_submesh_reverse,
        )
