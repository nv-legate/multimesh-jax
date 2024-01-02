from typing import List

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import config
from jax.experimental.pjit import AUTO
from jax.sharding import GSPMDSharding, Mesh, NamedSharding, PartitionSpec as P

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

        def arg_maker():
            def f():
                return (jnp.arange(16, dtype=np.float32).reshape(4, 4),)

            return jax.jit(f, out_shardings=(arg_shardings))()

        with mesh:
            self._test_against_untransformed(
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
            self._test_against_untransformed(
                c, arg_maker, arg_shardings=arg_shardings
            )

        legate.jax.unregister_task(r"(task\d+)")

    def test_fully_replicated_sharding(self):
        if jax.device_count() != 4:
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
            if isinstance(initial, GSPMDSharding):
                self.assertEqual(initial, final)

    def test_reshard_initial_final(self):
        if jax.device_count() != 8:
            self.skipTest("need 8 devices")

        logical_axes = [
            ("batch", "x"),
            ("embed", "y"),
        ]

        legate.jax.register_task(
            "embeddings",
            devices=[0, 1, 2, 3, 4, 5, 6, 7],
            dims=[8, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        legate.jax.register_task(
            "loss",
            devices=[0, 1, 2, 3, 4, 5, 6, 7],
            dims=[8, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        legate.jax.register_task(
            "layer0",
            devices=[0, 1, 2, 3, 4, 5, 6, 7],
            dims=[8, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        legate.jax.register_task(
            "layer1",
            devices=[0, 1, 2, 3, 4, 5, 6, 7],
            dims=[8, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        vocab = 2
        batch_size = 8
        seq = 1
        embed_size = 2

        def c(params, batch):
            def f(params, batch):
                embed, fc0, fc1, scale = params
                with jax.named_scope("embeddings"):
                    batch = legate.jax.with_sharding_constraint(
                        batch, P("batch", "seq")
                    )
                    embed = legate.jax.with_sharding_constraint(
                        embed, P("vocab", "embed")
                    )
                    batch = jax.nn.one_hot(batch, num_classes=vocab)
                    x = jnp.einsum("bsv,ve->bse", batch, embed)
                with jax.named_scope("layer0"):
                    fc0 = legate.jax.with_sharding_constraint(
                        fc0, P("embed-fc", "hidden")
                    )
                    x = legate.jax.with_sharding_constraint(
                        x, P("batch", "seq", "embed")
                    )
                    x = jnp.einsum("bsh,eh->bsh", x, fc0)
                with jax.named_scope("layer1"):
                    fc1 = legate.jax.with_sharding_constraint(
                        fc1, P("embed-fc", "hidden")
                    )
                    x = legate.jax.with_sharding_constraint(
                        x, P("batch", "seq", "embed")
                    )
                    x = jnp.einsum("bse,eh->bsh", x, fc1)
                with jax.named_scope("loss"):
                    x = legate.jax.with_sharding_constraint(
                        x, P("batch", "seq", "embed")
                    )
                    scale = legate.jax.with_sharding_constraint(
                        scale, P("batch", "seq", "embed")
                    )
                    loss = (x * scale).sum()
                    return loss

            return f(params, batch)

        mesh = Mesh(
            np.array(jax.devices()).reshape(8, 1, 1, 1, 1, 1, 1),
            ("batch", "model", "seq", "hidden", "vocab", "embed", "embed-fc"),
        )

        def make_lowered(mesh):
            with mesh:
                f = jax.jit(
                    c,
                    in_shardings=(AUTO(mesh), AUTO(mesh)),
                    out_shardings=(AUTO(mesh)),
                )
                batch = jax.core.ShapedArray((batch_size, seq), np.float32)
                fc0 = jax.core.ShapedArray(
                    (embed_size, embed_size), np.float32
                )
                fc1 = jax.core.ShapedArray(
                    (embed_size, embed_size), np.float32
                )
                embed = jax.core.ShapedArray((vocab, embed_size), np.float32)
                scale = jax.core.ShapedArray(
                    (batch_size, seq, embed_size), np.float32
                )
                params = (embed, fc0, fc1, scale)
                lowered = f.lower(params, batch).compile()
                return lowered

        lowered = make_lowered(mesh)
        reference_shardings = lowered.input_shardings[0]
        (
            _,
            fc0_sharding,
            fc1_sharding,
            scale_sharding,
        ) = lowered.input_shardings[0][0]

        replicated = NamedSharding(mesh, spec=P())
        batch_sharding = NamedSharding(
            mesh,
            spec=P(
                "batch",
            ),
        )
        self.assertEqual(fc0_sharding, replicated)
        self.assertEqual(fc1_sharding, replicated)
        self.assertEqual(scale_sharding, batch_sharding)

        logical_axes.append(("hidden", "x"))
        legate.jax.register_task(
            "layer0",
            devices=[0, 1, 2, 3],
            dims=[2, 2],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )
        legate.jax.register_task(
            "layer1",
            devices=[4, 5, 6, 7],
            dims=[2, 2],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        jax.clear_caches()

        lowered = make_lowered(mesh)
        arg_shardings = lowered.input_shardings[0]
        (
            _,
            fc0_sharding,
            fc1_sharding,
            scale_sharding,
        ) = lowered.input_shardings[0][0]
        batch_sharding = lowered.input_shardings[0][1]

        self.assertTrue(isinstance(fc0_sharding, GSPMDSharding))
        self.assertEqual(
            fc0_sharding._hlo_sharding.tile_assignment_devices(), [0, 1, 2, 3]
        )
        self.assertTrue(
            fc0_sharding._hlo_sharding.replicate_on_last_tile_dim()
        )

        self.assertTrue(isinstance(fc1_sharding, GSPMDSharding))
        self.assertEqual(
            fc1_sharding._hlo_sharding.tile_assignment_devices(), [4, 5, 6, 7]
        )
        self.assertTrue(
            fc1_sharding._hlo_sharding.replicate_on_last_tile_dim()
        )

        self.assertTrue(isinstance(scale_sharding, GSPMDSharding))
        self.assertEqual(
            scale_sharding._hlo_sharding.tile_assignment_devices(),
            [0, 1, 2, 3, 4, 5, 6, 7],
        )
        self.assertFalse(
            scale_sharding._hlo_sharding.replicate_on_last_tile_dim()
        )

        self.assertTrue(isinstance(batch_sharding, GSPMDSharding))
        self.assertEqual(
            batch_sharding._hlo_sharding.tile_assignment_devices(),
            [0, 1, 2, 3, 4, 5, 6, 7],
        )
        self.assertFalse(
            batch_sharding._hlo_sharding.replicate_on_last_tile_dim()
        )

        def arg_maker():
            def make_shape(shape, dtype=np.float32):
                size = np.prod(shape)
                return jnp.arange(size, dtype=dtype).reshape(shape)

            def f():
                batch = make_shape((batch_size, seq)) % vocab
                fc0 = make_shape((embed_size, embed_size))
                fc1 = make_shape((embed_size, embed_size))
                scale = make_shape((batch_size, seq, embed_size))
                embed = make_shape((vocab, embed_size))
                params = (embed, fc0, fc1, scale)
                return (params, batch)

            return f()

        with mesh:
            self._test_against_untransformed(
                c,
                arg_maker,
                arg_shardings=arg_shardings,
                reference_shardings=reference_shardings,
            )


if __name__ == "__main__":
    legate.jax.init()
    absltest.main(testLoader=jtu.JaxTestLoader())
