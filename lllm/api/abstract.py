import numpy as np
from jax import ShapedArray
from jax._src.api_util import _shaped_abstractify_handlers
from jax.core import pytype_aval_mappings as core_pytype_aval_mappings
from jax.interpreters.pxla import shard_arg_handlers
from jax.interpreters.xla import (
    canonicalize_dtype_handlers,
    pytype_aval_mappings,
)


class AbstractTensor:
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        if 0 in self.shape:
            raise Exception(f"huh? {shape}")

    def aval(self):
        return ShapedArray(self.shape, self.dtype)

    def __repr__(self):
        return f"AbstractTensor({self.shape},{self.dtype})"

    def __lt__(self, other):
        return self

    def __gt__(self, other):
        return self

    def astype(self, dtype):
        return AbstractTensor(self.shape, dtype)

    def expand_dims(self, axis):
        if isinstance(axis, int):
            axis = [axis]
        else:
            axis = list(axis)
        axis.sort(reverse=True)
        new_shape = list(self.shape)
        for ax in axis:
            new_shape.insert(ax, 1)
        print(self.shape, axis, new_shape)
        return AbstractTensor(new_shape, self.dtype)

    def __jax_array__(self):
        return self

    @property
    def ndim(self) -> int:
        return len(self.shape)


def _make_tensor_aval(tensor: AbstractTensor):
    return ShapedArray(tensor.shape, tensor.dtype)


_shaped_abstractify_handlers[AbstractTensor] = _make_tensor_aval
pytype_aval_mappings[AbstractTensor] = _make_tensor_aval
core_pytype_aval_mappings[AbstractTensor] = _make_tensor_aval

canonicalize_dtype_handlers[AbstractTensor] = lambda x: x


def _shard_tensor_aval(tensor: AbstractTensor, *args, **kwargs):
    return tensor


shard_arg_handlers[AbstractTensor] = _shard_tensor_aval


def make_abstract(*shape, dtype=np.float32):
    return AbstractTensor(shape, dtype)


def invoke_abstract(fun, *args, **kwargs):
    return fun(*args, **kwargs)
