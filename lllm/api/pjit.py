from contextlib import contextmanager
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax.core import Primitive
from jax.experimental import PartitionSpec as P
from jax.interpreters import ad, mlir
from jax.tree_util import tree_flatten, tree_unflatten

from lllm.key_utils import LegateKey
from lllm.mesh import LayerMesh


def i32_attr(i):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)


def _mark_logical_partition_impl(*args, **kwargs):
    raise Exception(
        "_mark_logical_partition_impl: should not be invoked."
        " mark_logical_partition should only occur inside jit"
    )


def _mark_logical_partition_lowering(ctx, x_node):
    op = mhlo.CustomCallOp(
        [x_node.type],
        [x_node],
        call_target_name=ir.StringAttr.get("Sharding"),
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(""),
        api_version=i32_attr(1),
        called_computations=ir.ArrayAttr.get([]),
        operand_layouts=None,
        result_layouts=None,
    )
    return op.results


mark_logical_partition_p = Primitive("mark_logical_partition")
mark_logical_partition_p.def_impl(_mark_logical_partition_impl)
mark_logical_partition_p.def_abstract_eval(lambda x: x)
mlir.register_lowering(
    mark_logical_partition_p, _mark_logical_partition_lowering
)


def mark_logical_partition_p_linear(ct, _, **kwargs):
    with jax.named_scope("partition_transpose"):
        return (mark_logical_partition_p.bind(ct, **kwargs),)


ad.deflinear2(mark_logical_partition_p, mark_logical_partition_p_linear)


def _mark_arg(pspec, arg):
    pspec = tuple(pspec)
    if len(pspec) < len(arg.shape):
        padding = len(arg.shape) - len(pspec)
        pspec = pspec + (None,) * padding
    with jax.named_scope(f"legate_axes={pspec}/"):
        return mark_logical_partition_p.bind(arg)


def _mark_logical_partition_on_pytree(args, axis_resources):
    if axis_resources is not None:
        axis_flat, _ = tree_flatten(axis_resources)
        flat_args, arg_treedef = tree_flatten(args)
        if len(axis_flat) != len(flat_args):
            raise Exception(
                "axis_resources={} does not match args of length {}".format(
                    str(axis_resources), len(flat_args)
                )
            )
        flat_marked_args = [
            _mark_arg(pspec, arg) for pspec, arg in zip(axis_flat, flat_args)
        ]
        return tree_unflatten(arg_treedef, flat_marked_args)
    return args


def _mark_gradient_scalar_impl(*args, **kwargs):
    raise Exception(
        "_mark_gradient_scalar_impl: should not be invoked. "
        "mark_gradient_scalar should only occur inside jit"
    )


def _mark_gradient_scalar_lowering(ctx, x_node):
    op = mhlo.CustomCallOp(
        [x_node.type],
        [x_node],
        call_target_name=ir.StringAttr.get("Sharding"),
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(""),
        api_version=i32_attr(1),
        called_computations=ir.ArrayAttr.get([]),
        operand_layouts=None,
        result_layouts=None,
    )
    return op.results


mark_gradient_scalar_p = Primitive("mark_gradient_scalar")
mark_gradient_scalar_p.def_impl(_mark_gradient_scalar_impl)
mark_gradient_scalar_p.def_abstract_eval(lambda x: x)
mlir.register_lowering(mark_gradient_scalar_p, _mark_gradient_scalar_lowering)


def mark_gradient_scalar_p_linear(ct, _, **kwargs):
    return (mark_gradient_scalar_p.bind(ct, **kwargs),)


ad.deflinear2(mark_gradient_scalar_p, mark_gradient_scalar_p_linear)


class LegatePjitFunction:
    def __init__(self, fun, in_axis_resources, out_axis_resources):
        self.in_axis_resources = in_axis_resources
        self.out_axis_resources = out_axis_resources
        self.fun = fun

    def __call__(self, *args):
        marked_args = _mark_logical_partition_on_pytree(
            args, self.in_axis_resources
        )
        outputs = self.fun(*marked_args)
        return _mark_logical_partition_on_pytree(
            outputs, self.out_axis_resources
        )


def pjit(
    fun,
    in_axis_resources=None,
    out_axis_resources=None,
    static_argnums=None,
    donate_argnums=(),
):
    wrapped_fun = LegatePjitFunction(
        fun, in_axis_resources, out_axis_resources
    )
    return jax.jit(
        wrapped_fun,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )


_last_scope_annotation = None
_next_task_id: Mapping[str, int] = {}


@contextmanager
def task(
    *,
    name: Optional[str] = None,
    replicate: bool = False,
    implicit_decomposition: bool = False,
    layer: Optional[int] = None,
    append=False,
    backprop=False,
):
    global _last_scope_annotation
    global _next_task_id
    if append:
        if _last_scope_annotation is None:
            raise Exception(
                "cannot append task context, no previous tasks exists"
            )
        scope_annotation = _last_scope_annotation
    elif replicate:
        # These should not get tagged with legate_key.
        # They should be pulled into all layers that use them.
        scope_annotation = "/replicate" + (
            "/" if name is None else f"/{name}/"
        )
    elif name is None:
        raise Exception(
            "Either name, append, or replicate must be given for task"
        )
    else:
        if layer is not None:
            name = LayerMesh.canonical_name(name, layer)

        if implicit_decomposition:
            task_id = None
        else:
            task_id = _next_task_id.get(name)
            if task_id is None:
                task_id = 0
                _next_task_id[name] = 1
            else:
                _next_task_id[name] += 1

        scope_annotation = LegateKey.metadata_string(
            name=name,
            task_id=task_id,
        )
        if implicit_decomposition:
            scope_annotation += "implicit_decomposition/"

    if backprop:
        scope_annotation += "legate_backprop/"

    _last_scope_annotation = scope_annotation
    with jax.named_scope(scope_annotation):
        yield


def with_sharding_constraint(arg, axis_resources):
    if not isinstance(axis_resources, P):
        axis_resources = P(*axis_resources)
    return _mark_logical_partition_on_pytree(arg, axis_resources)


def partition(arg, *axis_names):
    if len(axis_names) == 1 and isinstance(axis_names[0], P):
        raise Exception(
            "PartitionSpec passed to llm.partition, expected variadic args"
        )
    else:
        return _mark_logical_partition_on_pytree(arg, P(*axis_names))


def mark_gradient_scalar(arg):
    with jax.named_scope("legate_gradient_scalar"):
        return mark_gradient_scalar_p.bind(arg)


def reduce_body(fxn):
    def reduce_fxn(i: int, prev_result: Any, current_result: Any):
        flat_prev, tree = tree_flatten(prev_result)
        flat_current, _ = tree_flatten(current_result)
        new_result = []
        for prev, current in zip(flat_prev, flat_current):
            new_result.append(fxn(prev, current))
        result = tree_unflatten(tree, new_result)
        return result

    return reduce_fxn


def pytree_zeros(args: Any) -> Any:
    """Create an initial set of zeros that match the PyTree shape of the args

    Args:
        args (Any): The set of args to zero-match the shape of

    Returns:
        Any: The PyTree of zeros
    """
    flat_arg, tree = tree_flatten(args)
    flat_zeros = []
    for arg in flat_arg:
        flat_zeros.append(jnp.zeros(arg.shape, arg.dtype))
    return tree_unflatten(tree, flat_zeros)


def fori_reduce(
    lower: int,
    upper: int,
    args: Any,
    body_fun: Any,
    reduce: Any,
    reduce_init: Any,
    name: Optional[str] = None,
    implicit_decomposition: bool = False,
    unroll: bool = True,
) -> Any:
    """Runs code equivalent body_fun(i,args) for i in range(lower,upper) and
    reduces by invoking reduce(i, accum, results[i]) for i in the range.

    Args:
        lower (int): The lower bound of the loop range
        upper (int): The upper bound (exclusive) of the loop range
        args (Any): The args to pass to each loop of body_fun
        body_fun (Any): The function to invoke on each loop
        reduce (Any): The function invoked on each result returned by body_fun
        reduce_init (Any): The initial value for the reduce
        name (Optional[str]): A name for the reduce. Defaults to None.
        implicit_decomposition (boo): See implicit_decomposition documentation
                                      of task. Defaults to False.

    Returns:
        Any: the result of the reduction
    """

    if unroll:
        results = []
        for i in range(lower, upper):
            results.append(body_fun(i, args))
            result = reduce_init
            for idx, next_result in enumerate(results):
                result = reduce(idx + lower, result, next_result)
            return result
    else:

        def wrapped(i, val):
            old_result, args = val
            new_result = body_fun(i, args)
            with task(
                name=name, implicit_decomposition=implicit_decomposition
            ):
                result = jax.lax.cond(
                    i == 0,
                    lambda _: new_result,
                    lambda _: reduce(i, old_result, new_result),
                    None,
                )
            return result, args

        with jax.named_scope(f"microbatches={upper-lower}"):
            result, args = jax.lax.fori_loop(
                lower, upper, wrapped, (reduce_init, args)
            )
        return result
