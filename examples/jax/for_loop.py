import os
import sys

import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad
import jax
from jax.tree_util import tree_flatten, tree_unflatten

from lllm.api import (
    fori_reduce,
    mark_gradient_scalar,
    partition,
    pytree_zeros,
    reduce_body,
    task,
)

if len(sys.argv) > 1:
    xla_flags = sys.argv[1:]
else:
    xla_flags = [
        "--xla_dump_to=dump_microbatches",
        "--xla_dump_hlo_as_proto",
    ]
os.environ["XLA_FLAGS"] = " ".join(xla_flags)


def layer(lyr, params, x):
    mult_params, cos_params = params
    with task(name="layer", layer=lyr, backprop=True):
        x = partition(x, "x")
        mult_params = partition(mult_params, "x", "y")
        y = x.reshape(x.shape[0], 1) * mult_params
        cos_params = partition(cos_params, "x", "y")
        ein_xx = jnp.einsum("xy,Xy->xX", y, cos_params)
        ein_xx = partition(ein_xx, "x", "x")
        ein_xy = jnp.einsum("Xx,xy->Xy", ein_xx, mult_params)
        op_xxy = jnp.einsum("X,xy->Xxy", x, ein_xy)
        ein_xxy = jnp.einsum("xc,cXy->xXy", ein_xx, op_xxy)
        red_xx = jnp.einsum("xab,Xab->xX", op_xxy, ein_xxy)
        # shrink this down to 1-D so that the interface between
        # layers is much smaller than the work within a layer
        y = jnp.cos(red_xx).sum(axis=0)
        return partition(y, "x")


def loss_fn(params, x):
    """The provided loss function should drive the final layer
    parameters to mult_params = 0 and cos_params = pi/2
    This maximizes the value at cos(pi/2) = 0
    """
    for lyr, param in enumerate(params):
        x = layer(lyr, param, x)
    with task(
        append=True, backprop=True
    ):  # add the loss computation to the last layer
        return (x * x).sum()


def step_fn(params, batch, microbatch_size, num_microbatches):
    grad_fn = value_and_grad(loss_fn)

    def body_fun(i: int, args):
        batch, params = args
        offset = i * microbatch_size
        length = microbatch_size
        starts = [offset] + [0] * (batch.ndim-1)
        limits = [length] + list(batch.shape[1:])

        microbatch = jax.lax.dynamic_slice(batch, starts, limits)
        return grad_fn(params, microbatch)

    reduce = reduce_body(lambda x, y: x + y)
    reduce_init = (0, pytree_zeros(params))
    value, grads = fori_reduce(
        0,
        num_microbatches,
        name="microbatch",
        args=(batch, params),
        body_fun=body_fun,
        reduce_init=reduce_init,
        reduce=reduce,
        implicit_decomposition=True,
        unroll=False
    )

    flat_params, tree = tree_flatten(params)
    flat_grads, _ = tree_flatten(grads)
    new_params = []
    with task(name="update", implicit_decomposition=True):
        for param, grad in zip(flat_params, flat_grads):
            new_params.append(param - 0.05 * grad)
    return mark_gradient_scalar(value), tree_unflatten(tree, new_params)


def generate_input(*shape):
    return np.zeros(shape)


def run_step(num_layers: int = 4, num_microbatches: int = 4, size=1024):
    params = []
    microbatch_size = size // num_microbatches
    shape = (microbatch_size, size)
    for _ in range(num_layers):
        mult_params = generate_input(*shape)
        cos_params = generate_input(*shape)
        params.append((mult_params, cos_params))

    batch = generate_input(microbatch_size * num_microbatches)
    step = jit(step_fn, static_argnums=(2,3))
    loss, params = step(params, batch, microbatch_size, num_microbatches)
    print("Loss=", loss)


run_step()
