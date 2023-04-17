import os
import sys
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, value_and_grad
from jax.tree_util import tree_flatten, tree_unflatten

from lllm.api import mark_gradient_scalar, partition, task

if len(sys.argv) > 1:
    xla_flags = sys.argv[1:]
else:
    xla_flags = [
        "--xla_dump_to=dump_simple_layers",
        "--xla_dump_hlo_as_proto",
    ]
os.environ["XLA_FLAGS"] = " ".join(xla_flags)


def layer(lyr, params, x):
    mult_params, cos_params = params
    with task(name="layer", layer=lyr, backprop=True):
        x = partition(x, "x", "y")
        mult_params = partition(mult_params, "x", "y")
        y = x * mult_params
        cos_params = partition(cos_params, "x", "y")
        y = jnp.cos(y + cos_params)
        return partition(y, "x", "y")


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
        loss = mark_gradient_scalar((x * x).sum())
    return loss


def step_fn(params, x):
    value, grads = value_and_grad(loss_fn)(params, x)
    flat_grads, _ = tree_flatten(grads)
    flat_params, param_tree = tree_flatten(params)
    new_params = []
    with task(name="update", implicit_decomposition=True):
        for grad, param in zip(flat_grads, flat_params):
            param = partition(param, "x", "y")
            grad = partition(grad, "x", "y")
            new_param = partition(param - 0.05 * grad, "x", "y")
            new_params.append(new_param)
        return value, tree_unflatten(param_tree, new_params)


def run_step(num_layers: int = 2, shape: Tuple[int] = (16, 16)):
    params = []
    for lyr in range(num_layers):
        mult_params = np.random.randn(*shape)
        cos_params = np.random.randn(*shape)
        params.append((mult_params, cos_params))
    step = jit(step_fn)
    for i in range(1000):
        x = np.random.randn(*shape)
        loss, params = step(params, x)
    print("Loss=", loss)


run_step()
