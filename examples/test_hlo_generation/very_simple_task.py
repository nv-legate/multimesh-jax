import jax
import jax.numpy as jnp
import numpy as np

import legate.jax

print(jax.devices())

with legate.jax.ignore_transforms(False):

    def f(x):
        return x * x

    f = legate.jax.task(f, devices=jax.devices())

    def c(x):
        return f(f(x))

    def args_maker():
        return (jnp.arange(8, dtype=np.float32),)

    args = jax.jit(args_maker)()
    jf = jax.jit(c)
    res = jf(*args)
    print(res)
