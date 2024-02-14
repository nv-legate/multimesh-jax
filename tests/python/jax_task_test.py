import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import config
from jax.sharding import Mesh, PositionalSharding, PartitionSpec as P

import legate.jax
from legate.jax.test_util import LegateJaxTestCase

config.parse_flags_with_absl()


class TaskTest(LegateJaxTestCase):
    def make_shape(self, *shape, dtype=np.float32):
        size = np.prod(shape)
        return jnp.arange(size, dtype=dtype).reshape(shape)

    @jtu.sample_product(
        dtypes=[
            (np.float32,),
            (np.int16, np.float32),
            (np.int8, np.float64, np.int32),
        ],
    )
    def test_add_sequence(self, dtypes):
        if jax.device_count() != 1:
            self.skipTest("need a single device")

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

        self._test_against_reference(jnp_fxn, args_maker)

    def test_donation_single_argument(self):
        if jax.device_count() != 1:
            self.skipTest("need a single device")

        def c(x):
            def f(x):
                return x * x

            f = legate.jax.task(f)
            return f(x)

        def arg_maker():
            return (self.make_shape(4, 4),)

        self._test_against_reference(c, arg_maker, donate_argnums=(0,))

    def test_donation_multiple_arguments(self):
        if jax.device_count() != 1:
            self.skipTest("need a single device")

        def c(x, y, z):
            def f(x, y, z):
                return x * x, x * y, z

            f = legate.jax.task(f)

            def g(x, y, z):
                return x, x * y, z, x * z

            g = legate.jax.task(g)
            return g(*f(x, y, z))

        def arg_maker():
            x = self.make_shape(4, 4)
            y = self.make_shape(4, 4)
            z = self.make_shape(4, 4)
            return x, y, z

        self._test_against_reference(c, arg_maker, donate_argnums=(0, 2))

    def test_donation_sharded_arguments(self):
        if jax.device_count() != 2:
            self.skipTest("need 2 devices")

        def c(x, y):
            def f(x):
                return x * x

            f = legate.jax.task(f)

            def g(x, y):
                return x * x, x * y

            g = legate.jax.task(g)
            x = f(x)
            return g(x, y)

        def arg_maker():
            x = self.make_shape(4, 4)
            y = self.make_shape(4, 4)
            return x, y

        sharding = PositionalSharding(jax.devices()).reshape(2, 1)
        self._test_against_reference(
            c,
            arg_maker,
            donate_argnums=(0, 1),
            arg_shardings=(sharding, sharding),
        )
    
    def test_task_configure(self):
        if jax.device_count() != 4:
            self.skipTest("need 4 devices")        

        class Configurable:
            def __call__(self, *, iter: int):
                offset = iter * 2
                devices = np.array(jax.devices()[offset:offset+2]).reshape(2,1)
                return legate.jax.Task(
                    devices=devices,
                    device_axes=("x", "y"),
                    logical_axes=[("batch", "x")]
                )


        def c(x):
            def f(x, *, iter: int):
                x = legate.jax.with_sharding_constraint(x, P("batch", "model"))
                return x*x
            f = legate.jax.task(f, configure=Configurable(), configure_args=("iter",))

            for i in range(2):
                x = f(x, iter=i)
            loss = x.sum()
            return loss
    
        def arg_maker():
            return jnp.arange(8).reshape(4,2),

        sharding = PositionalSharding(jax.devices()).reshape(4, 1)
        mesh = Mesh(np.array(jax.devices()).reshape(4, 1), ("batch", "model"))
        with mesh:
            self._test_against_reference(
                c,
                arg_maker,
                arg_shardings=(sharding,)
            ) 


    def test_task_gradients(self):
        if jax.device_count() != 1:
            self.skipTest("need a single device")

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

        self._test_against_reference(jax.value_and_grad(c), arg_maker)


if __name__ == "__main__":
    legate.jax.init()
    absltest.main(testLoader=jtu.JaxTestLoader())
