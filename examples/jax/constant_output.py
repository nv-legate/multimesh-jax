import os
import sys
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit

from lllm.api import partition, task

if len(sys.argv) > 1:
    xla_flags = sys.argv[1:]
else:
    xla_flags = [
        "--xla_dump_to=dump_constant_output",
        "--xla_dump_hlo_as_proto",
    ]
os.environ["XLA_FLAGS"] = " ".join(xla_flags)

num_layers = 2


def step_fn(params, x):
    mult_params, cos_params = params
    with task(name="task", backprop=True):
        x = partition(x, "x", "y")
        mult_params = partition(mult_params, "x", "y")
        z = np.asarray([(1.0, 2.0), (3.0, 4.0)])
        w = np.asarray([5.0, 6.0])
        y = x * mult_params
        cos_params = partition(cos_params, "x", "y")
        y = jnp.cos(y + cos_params)
        return w, partition(y, "x", "y"), 1.0, x + 1.0, z.reshape(4, 1)


def run_step(shape: Tuple[int] = (1024, 1024)):
    x = np.random.randn(*shape)
    mult_params = np.random.randn(*shape)
    cos_params = np.random.randn(*shape)
    step = jit(step_fn)
    w, _, c, _, z = step((mult_params, cos_params), x)
    print("w=", w)
    print("c=", c)
    print("z=", z)


run_step()
