import jax
import jax.numpy as jnp
import numpy as np

from legate.jax import ignore_transforms, task

with ignore_transforms(False):

    def c(x, scale1, scale2):
        return x.sum() * scale1 * scale2

    c = task(c)

    def args_maker():
        arg = jnp.arange(8, dtype=np.float32)
        scale = 2
        return arg, scale, scale

    args = jax.jit(args_maker)()
    jf = jax.jit(c)
    res = jf(*args)
    print(res)
