import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import config, value_and_grad

from legate.jax import microbatch, register_task, task
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

        self._test_against_reference(c, args_maker)

    def test_microbatch_pre_post_task(self):
        def c(args):
            def m(args):
                x, y, z = args
                return (x + y + z).sum(axis=0)

            m = microbatch(m, dim=0, size=2)
            x, y, z = args
            x = 2 * x
            y = 2 * y
            z = 2 * z
            s = m((x, y, z))
            return s.sum()

        def args_maker():
            def make_shape(*shape):
                size = np.prod(shape)
                return jnp.arange(size).reshape(*shape)

            return ((make_shape(4, 4), make_shape(4, 1), make_shape(4, 1)),)

        self._test_against_reference(c, args_maker)

    def test_microbatch_implicit_pre_post_task(self):
        logical_axes = [
            ("batch", "x"),
            ("model", "y"),
        ]

        register_task(
            "layer0",
            devices=[0],
            dims=[1, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        register_task(
            "layer1",
            devices=[0],
            dims=[1, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        def c(args):
            @jax.jit
            def inner_comp(x, y):
                return x + y

            def m(args):
                x, y, z = args
                with jax.named_scope("layer0"):
                    s = inner_comp(x, y)
                with jax.named_scope("layer1"):
                    return (s + z).sum(axis=0)

            m = microbatch(m, dim=0, size=2)
            x, y, z = args
            with jax.named_scope("layer0"):
                x = 2 * x
                y = 2 * y
                z = 2 * z
            s = m((x, y, z))
            with jax.named_scope("layer1"):
                return s.sum()

        def args_maker():
            def make_shape(*shape):
                size = np.prod(shape)
                return jnp.arange(size, dtype=np.float32).reshape(*shape)

            return ((make_shape(4, 4), make_shape(4, 1), make_shape(4, 1)),)

        self._test_against_reference(c, args_maker)

    def test_microbatch_no_tasks(self):
        def c(args):
            def f(args):
                x, y, z = args
                return (x * y * z).sum()

            f = microbatch(f, dim=0, size=2)
            return f(args)

        def args_maker():
            def make_shape(*shape):
                size = np.prod(shape)
                return jnp.arange(size).reshape(*shape)

            return ((make_shape(4, 4), make_shape(4, 1), make_shape(4, 1)),)

        self._test_against_reference(c, args_maker)

    def test_multiple_microbatch_slices(self):
        def c(args):
            def f(args):
                x, y, z = args
                return (x * y * z).sum()

            f = microbatch(task(f), dim=0, size=2)
            return f(args)

        def args_maker():
            def make_shape(*shape):
                size = np.prod(shape)
                return jnp.arange(size).reshape(*shape)

            return ((make_shape(4, 4), make_shape(4, 1), make_shape(4, 1)),)

        self._test_against_reference(c, args_maker)

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

        self._test_against_reference(c, args_maker)

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

        self._test_against_reference(c, args_maker)

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

        self._test_against_reference(c, args_maker)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
