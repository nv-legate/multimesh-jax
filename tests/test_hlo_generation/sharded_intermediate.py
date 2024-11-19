import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

import legate.jax

with legate.jax.ignore_transforms(False):

    def f(x):
        return x + 2

    f = legate.jax.task(f)

    def g(x):
        return jnp.cos(x) * 2

    g = legate.jax.task(g)

    def c(x):
        return g(f(x))

    def args_maker():
        mesh = Mesh(np.array(jax.devices()), ("x",))
        sh = jax.sharding.NamedSharding(mesh, P("x"))
        return (
            jax.lax.with_sharding_constraint(
                jnp.arange(8, dtype=np.float32), sh
            ),
        )

    args = jax.jit(args_maker)()
    jf = jax.jit(c)
    res = jf(*args)
    print(res)
