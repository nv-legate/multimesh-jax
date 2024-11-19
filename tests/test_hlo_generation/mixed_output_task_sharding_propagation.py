import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

import legate.jax

spec = P(
    "x",
)
mesh = Mesh(np.array(jax.devices()), ("x",))
sh = jax.sharding.NamedSharding(mesh, spec)
replicated = jax.sharding.NamedSharding(mesh, P())


with legate.jax.ignore_transforms(False):

    def f(x, scale):
        return x * x, x + scale

    f = legate.jax.task(f)

    def g(x, y, z):
        return (jnp.cos(x) + y) * z

    g = legate.jax.task(g)

    def c(x, scale):
        a, b = f(x, scale)
        return g(a, b, x), a, b

    def args_maker():
        arg = jnp.arange(8, dtype=np.float32)
        scale = 2
        return arg, scale

    args = jax.jit(args_maker, out_shardings=(sh, replicated))()
    res = jax.jit(c, out_shardings=(sh, sh, sh))(*args)
    print(res)
