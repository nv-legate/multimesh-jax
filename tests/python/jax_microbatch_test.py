import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import config, value_and_grad

from legate.jax import microbatch, task
from legate.jax.test_util import LegateJaxTestCase

config.parse_flags_with_absl()


class MicrobatchTest(LegateJaxTestCase):
    def _args_maker(self, shape):
        return jnp.arange(np.prod(shape)).reshape(shape)

    def test_simple_microbatch(self):
        def c(x):
            def f(x):
                return (x * x).sum()

            f = microbatch(task(f), dim=0, size=2)
            return f(x)

        def args_maker():
            return (jnp.arange(4),)

        self._test_against_native(c, args_maker)

    def test_multiple_microbatch(self):
        def c(x, param1, param2):
            def g(x, param):
                return (x * param).sum()

            g = task(g, devices=jax.devices())

            def f(x, param):
                return x * param

            f = task(f, devices=jax.devices())

            def residual(x, param1, param2):
                x = f(x, param1)
                return g(x, param2)

            residual = microbatch(residual, dim=0, size=2)

            return residual(x, param1, param2)

        def args_maker():
            x = jnp.arange(16).reshape(4, 4)
            p1 = jnp.arange(4)
            p2 = jnp.arange(4)
            return x, p1, p2

        self._test_against_native(c, args_maker)

    def test_simple_microbatch_grad(self):
        def c(param, x):
            def f(param, x):
                return (x * param).sum()

            f = value_and_grad(task(f, devices=jax.devices()))
            f = microbatch(f, argnum=1, dim=0, size=2)
            return f(param, x)

        def args_maker():
            p = jnp.arange(4, dtype=np.float32)
            x = jnp.arange(16, dtype=np.float32).reshape(4, 4)
            return p, x

        self._test_against_native(c, args_maker)

    def test_multiple_microbatch_grad(self):
        def c(params, x):
            def g(x, param):
                return (x * param).sum()

            g = task(g, devices=jax.devices())

            def f(x, param):
                return x * param

            f = task(f, devices=jax.devices())

            def residual(params, x):
                param1, param2 = params
                x = f(x, param1)
                return g(x, param2)

            residual = microbatch(
                value_and_grad(residual), argnum=1, dim=0, size=2
            )

            return residual(params, x)

        def args_maker():
            p1 = jnp.arange(4, dtype=np.float32)
            p2 = jnp.arange(4, dtype=np.float32)
            x = jnp.arange(16, dtype=np.float32).reshape(4, 4)
            return (p1, p2), x

        self._test_against_native(c, args_maker)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
