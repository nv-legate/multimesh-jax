import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import config

import legate.jax
from legate.jax.test_util import LegateJaxTestCase

config.parse_flags_with_absl()


class TaskTest(LegateJaxTestCase):
    @jtu.sample_product(
        dtypes=[
            (np.float32,),
            (np.int16, np.float32),
            (np.int8, np.float64, np.int32),
        ],
    )
    def test_add_sequence(self, dtypes):
        shape = [4, 4]

        def args_maker():
            return [
                jnp.arange(np.prod(shape), dtype=dtype).reshape(shape)
                for dtype in dtypes
            ]

        def f(lhs, rhs):
            return [l + r for (l, r) in zip(lhs, rhs)]

        def jnp_fxn(*arrs):
            res = arrs[:]
            for _ in range(3):
                res = legate.jax.task(f)(res, arrs)
            return res

        self._test_against_cuda(jnp_fxn, args_maker)

    def test_task_gradients(self):
        def arg_maker():
            return jnp.arange(8, dtype=np.float32), 2

        def c(x, scale):
            def f(x, scale):
                return x * x * scale

            f = legate.jax.task(f)

            def g(x, y):
                return (jnp.cos(x) + y).sum()

            g = legate.jax.task(g)

            return g(f(x, scale), x)

        self._test_against_cuda(jax.value_and_grad(c), arg_maker)


if __name__ == "__main__":
    legate.jax.init()
    absltest.main(testLoader=jtu.JaxTestLoader())
