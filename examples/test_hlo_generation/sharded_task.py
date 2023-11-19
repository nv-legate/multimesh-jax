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
        return x * x * scale

    f = legate.jax.task(f, sharding=(sh, replicated))

    def g(x, y):
        return (jnp.cos(x) + y).sum()

    g = legate.jax.task(g, sharding=(sh, replicated))

    def c(x, scale):
        return g(f(x, scale), x)

    def args_maker():
        arg = jnp.arange(8, dtype=np.float32)
        scale = 2
        return arg, scale

    args = jax.jit(args_maker, out_shardings=(sh, None))()
    jf = jax.jit(
        jax.value_and_grad(c),
        in_shardings=(
            sh,
            replicated,
        ),
        out_shardings=(
            replicated,
            sh,
        ),
    )
    loss, grads = jf(*args)
    print(loss)
    print(grads)
