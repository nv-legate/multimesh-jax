import jax
import jax.numpy as jnp
import numpy as np

from legate.jax import ignore_transforms, task

with ignore_transforms(False):

    def f(x, scale):
        return x * x * scale

    f = task(f, devices=jax.devices())

    def g(x, y):
        return (jnp.cos(x) + y).sum()

    g = task(g, devices=jax.devices())

    def c(x, scale):
        x = 2 * x
        return g(f(x, scale), x)

    def args_maker():
        arg = jnp.arange(8, dtype=np.float32)
        scale = 2
        return arg, scale

    args = jax.jit(args_maker)()
    jf = jax.jit(c)
    res = jf(*args)
    print(res)
