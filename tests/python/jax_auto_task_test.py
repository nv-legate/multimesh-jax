from typing import List

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import config
from jax.experimental.pjit import AUTO
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import legate.jax
from legate.jax.test_util import LegateJaxTestCase

config.parse_flags_with_absl()


class TaskTest(LegateJaxTestCase):
    def test_register_task(self):
        if jax.device_count() != 2:
            self.skipTest("need 2 devices")

        logical_axes = [
            ("batch", "x"),
            ("model", "y"),
        ]
        legate.jax.register_task(
            "task0",
            devices=[0, 1],
            dims=[2, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        legate.jax.register_task(
            "task1",
            devices=[0, 1],
            dims=[2, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        def c(x):
            with jax.named_scope("task0"):
                x = legate.jax.with_sharding_constraint(x, P("batch", "model"))
                x = x * jnp.sin(x)
                x = legate.jax.with_sharding_constraint(x, P("batch", "model"))
            with jax.named_scope("task1"):
                x = x * x
                return legate.jax.with_sharding_constraint(x, P("batch", None))

        def test(x):
            return c(x)

        mesh = Mesh(np.array(jax.devices()).reshape(2, 1), ("batch", "model"))
        with mesh:
            f = jax.jit(
                c, in_shardings=(AUTO(mesh),), out_shardings=(AUTO(mesh))
            )
            aval = jax.core.ShapedArray((4, 4), np.float32)
            lowered = f.lower(aval).compile()

        test_sharding = NamedSharding(mesh, P("batch", "model"))
        self.assertTrue(
            test_sharding.is_equivalent_to(lowered.input_shardings[0][0], 2)
        )

        arg_shardings = lowered.input_shardings[0]
        print("ARG=", arg_shardings)

        def arg_maker():
            def f():
                return (jnp.arange(16, dtype=np.float32).reshape(4, 4),)

            return jax.jit(f, out_shardings=(arg_shardings))()

        with mesh:
            self._test_against_native(
                test, arg_maker, arg_shardings=arg_shardings
            )

    def tearDown(self):
        legate.jax.clear_tasks()

    def test_register_factory(self):
        if jax.device_count() != 2:
            self.skipTest("need 2 devices")

        logical_axes = [
            ("batch", "x"),
            ("model", "y"),
        ]

        def device_factory(name: str) -> List[int]:
            return [0, 1]

        legate.jax.register_task_factory(
            r"(task\d+)",
            device_callback=device_factory,
            dims=[2, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        def c(x):
            with jax.named_scope("task0"):
                x = legate.jax.with_sharding_constraint(x, P("batch", "model"))
                x = x * jnp.sin(x)
                x = legate.jax.with_sharding_constraint(x, P("batch", "model"))
            with jax.named_scope("task1"):
                x = x * x
                return legate.jax.with_sharding_constraint(x, P("batch", None))

        mesh = Mesh(np.array(jax.devices()).reshape(2, 1), ("batch", "model"))
        with mesh:
            f = jax.jit(
                c, in_shardings=(AUTO(mesh),), out_shardings=(AUTO(mesh))
            )
            aval = jax.core.ShapedArray((4, 4), np.float32)
            lowered = f.lower(aval).compile()

        test_sharding = NamedSharding(mesh, P("batch", "model"))
        self.assertTrue(
            test_sharding.is_equivalent_to(lowered.input_shardings[0][0], 2)
        )

        arg_shardings = lowered.input_shardings[0]

        def arg_maker():
            def f():
                return (jnp.arange(16, dtype=np.float32).reshape(4, 4),)

            return jax.jit(f, out_shardings=(arg_shardings))()

        with mesh:
            self._test_against_native(
                c, arg_maker, arg_shardings=arg_shardings
            )

        legate.jax.unregister_task(r"(task\d+)")

    def test_fully_replicated_sharding(self):
        if jax.device_count() < 4:
            self.skipTest("need 4 devices")

        logical_axes = [
            ("batch", "x"),
            ("model", "y"),
        ]

        legate.jax.register_task(
            "task0",
            devices=[0, 1],
            dims=[2, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        legate.jax.register_task(
            "task1",
            devices=[2, 3],
            dims=[2, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        def c(x, y):
            with jax.named_scope("task0"):
                x = legate.jax.with_sharding_constraint(x, P("batch", "model"))
                x = x * jnp.sin(x)
                x = legate.jax.with_sharding_constraint(x, P("batch", "model"))
            with jax.named_scope("task1"):
                y = legate.jax.with_sharding_constraint(y, P(None, None))
                x = x * y
                return legate.jax.with_sharding_constraint(x, P("batch", None))

        mesh = Mesh(np.array(jax.devices()).reshape(4, 1), ("batch", "model"))
        with mesh:
            f = jax.jit(
                c,
                in_shardings=(AUTO(mesh), AUTO(mesh)),
                out_shardings=(AUTO(mesh)),
            )
            aval = jax.core.ShapedArray((4, 4), np.float32)
            lowered = f.lower(aval, aval).compile()

        arg_shardings = lowered.input_shardings[0]
        with mesh:
            f = jax.jit(c, in_shardings=arg_shardings)
            aval = jax.core.ShapedArray((4, 4), np.float32)
            lowered = f.lower(aval, aval).compile()

        post_arg_shardings = lowered.input_shardings[0]

        for initial, final in zip(arg_shardings, post_arg_shardings):
            self.assertEqual(initial, final)


if __name__ == "__main__":
    legate.jax.init()
    absltest.main(testLoader=jtu.JaxTestLoader())
