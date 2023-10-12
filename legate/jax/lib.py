import jax
from jax.tree_util import tree_flatten, tree_unflatten

from .no_op import mark_gradient, mark_loss


def value_and_grad(fxn):
    jf = jax.value_and_grad(fxn)

    def wrapped(*args, **kwargs):
        value, grads = jf(*args, **kwargs)
        new_value = mark_loss(value)
        flat_grads, tree = tree_flatten(grads)
        new_grads = [mark_gradient(g) for g in flat_grads]
        return new_value, tree_unflatten(tree, new_grads)

    return wrapped


def init():
    jax.value_and_grad = value_and_grad
