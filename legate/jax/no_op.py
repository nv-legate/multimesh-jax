from collections.abc import Iterable
from functools import partial
from typing import Optional

import jax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import mhlo
from jax.core import Primitive
from jax.interpreters import ad, mlir


def _no_op_impl(*args, **kwargs):
    raise Exception(
        "no_op should not be invoked outside jit, "
        "this is for marking operations for jit"
    )


def i32_attr(i):
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), i)


def no_op_lowering(
    ctx, *args, abstract, target: str, config: Optional[str] = None
):
    config = "" if config is None else config
    result = abstract(*args)
    result = result if isinstance(result, Iterable) else [result]

    op = mhlo.CustomCallOp(
        [x.type for x in result],
        [x for x in args],
        call_target_name=ir.StringAttr.get(target),
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(config),
        api_version=i32_attr(1),
        called_computations=ir.ArrayAttr.get([]),
        operand_layouts=None,
        result_layouts=None,
    )
    return op.results


def no_op(name: str, config: Optional[str] = None):
    no_op_p = Primitive(name)
    no_op_p.def_impl(_no_op_impl)
    if config is None:
        def abstract(x, config=None):
            return x
    else:
        def abstract(x):
            return x

    no_op_p.def_abstract_eval(abstract)
    if config is None:
        mlir.register_lowering(
            no_op_p,
            partial(no_op_lowering, abstract=abstract, target=name),
        )
    else:        
        mlir.register_lowering(
            no_op_p,
            partial(no_op_lowering, abstract=abstract, target=name, config=config),
        )

    def no_op_p_linear(ct, _, **kwargs):
        return (no_op_p.bind(ct, **kwargs),)

    ad.deflinear2(no_op_p, no_op_p_linear)

    if config is None:
        def wrapped(*args, config=None, **kwargs):
            with jax.named_scope(name):
                return no_op_p.bind(*args, **kwargs, config=config)

        return wrapped
        
    def wrapped(*args, **kwargs):
        with jax.named_scope(name):
            return no_op_p.bind(*args, **kwargs)

    return wrapped


mark_gradient = no_op(name="Marking", config="gradient")
mark_loss = no_op(name="Marking", config="loss")
