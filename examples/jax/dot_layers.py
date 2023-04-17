import os
import sys

import numpy as np
from jax import jit, value_and_grad
from jax.tree_util import tree_flatten, tree_unflatten

from lllm.api import mark_gradient_scalar, partition, task

if len(sys.argv) > 1:
    xla_flags = sys.argv[1:]
else:
    xla_flags = [
        "--xla_dump_to=dump_dot_layers",
        "--xla_dump_hlo_as_proto",
    ]
os.environ["XLA_FLAGS"] = " ".join(xla_flags)

num_layers = 2


def layer(lyr, params, x):
    dot_params, bias_params = params
    with task(name="layer", layer=lyr, backprop=True):
        x = partition(x, "batch", "x")
        dot_params = partition(dot_params, "x", "z")
        y = x.dot(dot_params)
        bias_params = partition(bias_params, "z")
        y = y + bias_params
        return partition(y, "batch", "z")


def loss_fn(params, x, truth):
    for lyr, param in enumerate(params):
        x = layer(lyr, param, x)
    # add the loss computation to the most recent layer
    with task(append=True):
        delta = x - truth
        loss = (delta * delta).sum()
        return mark_gradient_scalar(loss)


def step_fn(params, x, truth):
    value, grads = value_and_grad(loss_fn)(params, x, truth)
    flat_grads, _ = tree_flatten(grads)
    flat_params, param_tree = tree_flatten(params)
    with task(name="update", implicit_decomposition=True):
        new_params = [
            (param - 0.05 * grad)
            for grad, param in zip(flat_grads, flat_params)
        ]
        return value, tree_unflatten(param_tree, new_params)


def run_step(num_layers: int = 2, batch_size: int = 16, hidden_size=32):
    params = []
    for lyr in range(num_layers):
        dot_params = np.random.randn(hidden_size, hidden_size)
        bias_params = np.random.randn(hidden_size)
        params.append((dot_params, bias_params))
    step = jit(step_fn)
    x = np.random.randn(batch_size, hidden_size)
    truth = np.random.randn(hidden_size)
    loss, params = step(params, x, truth)
    print("Loss=", loss)


run_step()
