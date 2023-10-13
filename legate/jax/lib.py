import atexit
from contextlib import contextmanager

import jax
from jax import value_and_grad as jax_value_and_grad
from jax.tree_util import tree_flatten, tree_unflatten

from .legate_jax_impl import shutdown
from .no_op import mark_gradient, mark_loss

_ignore_transforms = 0


@contextmanager
def ignore_transforms():
    global _ignore_transforms
    _ignore_transforms += 1
    yield
    _ignore_transforms -= 1


def should_ignore_transforms() -> bool:
    backend = jax._src.xla_bridge.get_backend()
    if _ignore_transforms > 0:
        return True

    backend = jax._src.xla_bridge.get_backend()
    return backend.platform != "legate"


def value_and_grad(fxn):
    jf = jax_value_and_grad(fxn)
    if should_ignore_transforms():
        return jf

    def wrapped(*args, **kwargs):
        value, grads = jf(*args, **kwargs)
        new_value = mark_loss(value)
        flat_grads, tree = tree_flatten(grads)
        new_grads = [mark_gradient(g) for g in flat_grads]
        return new_value, tree_unflatten(tree, new_grads)

    return wrapped


def init():
    # TODO: I don't think it is required anymore to mark
    # gradients and scalars. The new tasking system should
    # just work. Verify before removing this comment.
    # jax.value_and_grad = value_and_grad
    atexit.register(shutdown)
