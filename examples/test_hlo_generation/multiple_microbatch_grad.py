from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad

from legate.jax import ignore_transforms, microbatch, task

with ignore_transforms(False):

    def g(x, param):
        return (x * param).sum()

    g = task(g, devices=jax.devices())

    def f(x, param):
        return x * param

    f = task(f, devices=jax.devices())

    def residual(x, param1, param2):
        x = f(x, param1)
        return g(x, param2)

    def c(params, x):
        param1, param2 = params
        return residual(x, param1, param2)

    c = microbatch(value_and_grad(c), argnum=1, dim=0, size=2)

    @partial(jax.jit, static_argnums=0)
    def args_maker(shape):
        n = np.prod(shape)
        arg = jnp.arange(n, dtype=np.float32).reshape(shape)
        return arg

    batch = args_maker((4, 4))
    params1 = args_maker((4,))
    params2 = args_maker((4,))
    jf = jax.jit(c)
    res = jf((params1, params2), batch)
    print(res)
