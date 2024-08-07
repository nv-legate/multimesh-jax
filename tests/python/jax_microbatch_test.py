import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import config, value_and_grad
from jax.experimental.pjit import AUTO
from jax.sharding import Mesh, PartitionSpec as P

from legate.jax import (
    enable_task_fusion,
    microbatch,
    register_task,
    reset as lj_reset,
    task,
    with_sharding_constraint,
)
from legate.jax.test_util import LegateJaxTestCase

config.parse_flags_with_absl()


class MicrobatchTest(LegateJaxTestCase):
    def _args_maker(self, shape):
        return jnp.arange(np.prod(shape)).reshape(shape)

    def test_simple_microbatch(self):
        if jax.device_count() != 1:
            self.skipTest("need 1 device")

        def c(x):
            def f(x):
                return (x * x).sum()

            f = microbatch(task(f), dim=0, size=2)
            return f(x)

        def args_maker():
            return (jnp.arange(4),)

        self._test_against_reference(c, args_maker)

    def test_microbatch_pre_post_task(self):
        if jax.device_count() != 1:
            self.skipTest("need 1 device")

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

    def test_microbatch_1f1b(self):
        if jax.device_count() != 1:
            self.skipTest("need 1 device")

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
        register_task(
            "layer2",
            devices=[0],
            dims=[1, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        register_task(
            "layer3",
            devices=[0],
            dims=[1, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        register_task(
            "final",
            devices=[0],
            dims=[1, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        def c(params, batch):
            def inner_comp(x, y):
                return jnp.einsum("ac,cb->ab", x, y)

            def m(params, batch):
                x, y, z, w = params
                with jax.named_scope("layer0"):
                    s = inner_comp(batch, x)
                with jax.named_scope("layer1"):
                    s = inner_comp(s, y)
                with jax.named_scope("layer2"):
                    s = inner_comp(s, z)
                with jax.named_scope("layer3"):
                    s = inner_comp(s, w)
                    return s.sum()

            g = value_and_grad(m)
            g = microbatch(
                g, dim=0, size=2, argnum=1, schedule="1f1b", num_stages=4
            )
            return g(params, batch)

        def args_maker():
            def make_shape(*shape):
                size = np.prod(shape)
                return jnp.arange(size, dtype=np.float32).reshape(*shape)

            return (
                (
                    make_shape(4, 4),
                    make_shape(4, 4),
                    make_shape(4, 4),
                    make_shape(4, 4),
                ),
                make_shape(12, 4),
            )

        with enable_task_fusion(False):
            self._test_against_reference(c, args_maker)

    def test_microbatch_implicit_pre_post_task(self):
        if jax.device_count() != 1:
            self.skipTest("need 1 device")

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

    def test_microbatch_implicit_dynamic_slice_devices(self):
        if jax.device_count() != 4:
            self.skipTest("need 4 devices")

        def c(batch, params):
            @jax.jit
            def inner_comp(x, y):
                return jnp.einsum("ab,bc->ac", x, y)

            def m(batch, params):
                x, y, z = params
                x = with_sharding_constraint(x, P("model", None))
                y = with_sharding_constraint(y, P("model", None))
                z = with_sharding_constraint(z, P("model", None))
                batch = with_sharding_constraint(batch, P("batch", "model"))
                with jax.named_scope("layer0"):
                    s = inner_comp(batch, x)
                with jax.named_scope("layer1"):
                    s = inner_comp(s, y)
                with jax.named_scope("layer2"):
                    return inner_comp(s, z).sum(axis=0)

            m = microbatch(m, dim=0, size=2)

            s = m(batch, params)
            with jax.named_scope("layer2"):
                return s.sum()

        def args_maker(abstract: bool = False):
            def make_shape(*shape):
                if abstract:
                    return jax.core.ShapedArray(shape, np.float32)
                size = np.prod(shape)
                return jnp.arange(size, dtype=np.float32).reshape(*shape)

            return (
                make_shape(4, 4),
                (make_shape(4, 1), make_shape(4, 1), make_shape(4, 1)),
            )

        def make_lowered(mesh):
            with mesh:
                f = jax.jit(
                    c,
                    in_shardings=(
                        AUTO(mesh),
                        (AUTO(mesh), AUTO(mesh), AUTO(mesh)),
                    ),
                    out_shardings=(AUTO(mesh)),
                )
                batch, params = args_maker(abstract=True)
                print(batch, params)
                lowered = f.lower(batch, params).compile()
                return lowered

        mesh = Mesh(
            np.array(jax.devices()).reshape(1, 4),
            ("batch", "model"),
        )
        lowered = make_lowered(mesh)
        reference_shardings = lowered.input_shardings[0][0]

        logical_axes = [
            ("batch", "x"),
            ("model", "y"),
        ]

        register_task(
            "layer0",
            devices=[0, 1, 2, 3],
            dims=[1, 2],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
            loop_submesh_size=2,
        )
        register_task(
            "layer1",
            devices=[0, 1],
            dims=[1, 2],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        register_task(
            "layer2",
            devices=[2, 3],
            dims=[1, 2],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        jax.clear_caches()

        lowered = make_lowered(mesh)
        arg_shardings = lowered.input_shardings[0]

        with mesh:
            self._test_against_reference(
                c,
                args_maker,
                arg_shardings=arg_shardings,
                reference_shardings=reference_shardings,
            )

    def test_microbatch_no_tasks(self):
        if jax.device_count() != 1:
            self.skipTest("need 1 device")

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
        if jax.device_count() != 1:
            self.skipTest("need 1 device")

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
        if jax.device_count() != 1:
            self.skipTest("need 1 device")

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
        if jax.device_count() != 1:
            self.skipTest("need 1 device")

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

        with enable_task_fusion(False):
            self._test_against_reference(c, args_maker)

    def test_multiple_microbatch_grad(self):
        if jax.device_count() != 1:
            self.skipTest("need 1 device")

        lj_reset()

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
