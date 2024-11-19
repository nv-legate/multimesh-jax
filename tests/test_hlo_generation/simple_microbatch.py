from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from legate.jax import ignore_transforms, microbatch, task

with ignore_transforms(False):

    def f(x):
        return (x * x).sum()

    f = task(f, devices=jax.devices())
    f = microbatch(f, dim=0, size=2)

    @partial(jax.jit, static_argnums=0)
    def args_maker(shape):
        n = np.prod(shape)
        arg = jnp.arange(n, dtype=np.float32).reshape(shape)
        return arg

    batch = args_maker((4))
    jf = jax.jit(f)
    res = jf(batch)
    print(res)
