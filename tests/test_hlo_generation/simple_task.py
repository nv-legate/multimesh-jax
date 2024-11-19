import jax
import jax.numpy as jnp
import numpy as np

import legate.jax

with legate.jax.ignore_transforms(False):

    def f(x, scale):
        return x * x * scale

    f = legate.jax.task(f, devices=jax.devices())

    def g(x, y):
        return (jnp.cos(x) + y).sum()

    g = legate.jax.task(g, devices=jax.devices())

    def c(x, scale):
        return g(f(x, scale), x)

    def args_maker():
        arg = jnp.arange(8, dtype=np.float32)
        scale = 2
        return arg, scale

    args = jax.jit(args_maker)()
    jf = jax.jit(jax.value_and_grad(c))
    res = jf(*args)
    print(res)
