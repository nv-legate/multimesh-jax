import os
import sys

import jax.numpy as jnp
import numpy as np
from jax import grad
from jax.experimental import PartitionSpec as P

from lllm.api import partition, pjit, task

if len(sys.argv) > 1:
    xla_flags = sys.argv[1:]
else:
    xla_flags = [
        "--xla_dump_to=dump_replicate",
        "--xla_dump_hlo_as_proto",
    ]
os.environ["XLA_FLAGS"] = " ".join(xla_flags)


def loss(params, x):
    mult_params, cos_params = params
    with task(name="scale", replicate=True):
        scale = jnp.arange(x.size).reshape(x.shape)
    with task(name="mult"):
        x = partition(x, "x", "y")
        a = scale * mult_params * x + 1
    with task(name="cos"):
        b = scale * jnp.cos(a * cos_params)
        return b.sum()


def test(x, mult_params, cos_params):
    mult_grads, cos_grads = grad(loss)((mult_params, cos_params), x)
    with task(name="update", implicit_decomposition=True):
        mult_params += mult_grads
        cos_params += cos_grads
        return mult_params, cos_params


inp = jnp.arange(8).astype(np.float32).reshape(4, 2)
mult_params = jnp.arange(8).astype(np.float32).reshape(4, 2)
cos_params = jnp.arange(8).astype(np.float32).reshape(4, 2)

pspec = P("x", "y")
in_pjit = (pspec, (pspec, pspec))
out_pjit = (pspec, pspec)

res = pjit(test, in_axis_resources=in_pjit, out_axis_resources=out_pjit)(
    inp, mult_params, cos_params
)
print(res)
