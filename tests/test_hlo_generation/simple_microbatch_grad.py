from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

from legate.jax import ignore_transforms, microbatch, task

with ignore_transforms(True):

    def f(param, x):
        return (x * param).sum()

    f = value_and_grad(task(f, devices=jax.devices()))
    f = microbatch(f, argnum=1, dim=0, size=2)

    @partial(jax.jit, static_argnums=0)
    def args_maker(shape):
        n = np.prod(shape)
        arg = jnp.arange(n, dtype=np.float32).reshape(shape)
        return arg

    batch = args_maker((4))
    scalar = jnp.ones((1,)) * 2
    jf = jax.jit(f)
    res = jf(scalar, batch)
    print(res)
