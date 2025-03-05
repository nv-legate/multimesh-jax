from jax_plugins.legate import init_test

init_test()

from typing import List

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import config
from jax.experimental.custom_partitioning import custom_partitioning
from jax.experimental.pjit import AUTO
from jax.lax import with_sharding_constraint
from jax.sharding import (
    GSPMDSharding,
    Mesh,
    NamedSharding,
    PartitionSpec as P,
    PositionalSharding,
)

import legate.jax
from legate.jax import MeshWrapper
from legate.jax.test_util import LegateJaxTestCase

config.parse_flags_with_absl()


def make_shape(*shape, dtype=np.float32):
    size = np.prod(shape)
    return jnp.arange(size, dtype=dtype).reshape(shape)


class TaskTest(LegateJaxTestCase):
    def _test_register_task(self):
        def c(x):
            with jax.named_scope("task0"):
                x = with_sharding_constraint(x, P("batch", "model"))
                x = x * jnp.sin(x)
                x = with_sharding_constraint(x, P("batch", "model"))
            with jax.named_scope("task1"):
                x = x * x
                return with_sharding_constraint(x, P("batch", None))

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
            self._test_against_reference(
                test, arg_maker, arg_shardings=arg_shardings
            )

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
        self._test_register_task()

    def test_register_task_context(self):
        if jax.device_count() != 2:
            self.skipTest("need 2 devices")

        with legate.jax.context(autoshard=True):
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
            self._test_register_task()

    def test_only_fuse_loop_tasks(self):
        if jax.device_count() != 2:
            self.skipTest("need 2 devices")

        with legate.jax.context(autoshard=True):
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

            with legate.jax.only_fuse_loop_tasks(True):
                self._test_register_task()
            with legate.jax.only_fuse_loop_tasks(False):
                self._test_register_task()

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

        legate.jax.register_task(
            r"(task\d+)",
            callback=device_factory,
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
            self._test_against_reference(
                c, arg_maker, arg_shardings=arg_shardings
            )

        legate.jax.clear_tasks()

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
        with mesh, legate.jax.autoshard(True):
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

    def test_reshard_argument(self):
        self.skipTest(
            "do not yet support resharding across different mesh sizes"
        )
        if jax.device_count() != 4:
            self.skipTest("need 4 devices")

        mesh = Mesh(
            np.array(jax.devices()).reshape(4, 1, 1),
            ("batch", "cxn", "ext"),
        )

        logical_axes = [
            ("batch", "x"),
            ("embed", "x"),
            ("model", "y"),
        ]

        legate.jax.register_task(
            "layer0",
            devices=[0, 1, 2, 3],
            dims=[4, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        legate.jax.register_task(
            "layer1",
            devices=[0, 1],
            dims=[2, 1],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        def c(batch, params):
            def f(batch, params):
                (batch0, batch1) = batch
                (fc0, fc1) = params
                with jax.named_scope("layer0"):
                    batch = legate.jax.with_sharding_constraint(
                        batch0, P("batch", "cxn")
                    )
                    fc0 = legate.jax.with_sharding_constraint(
                        fc0, P("cxn", "ext")
                    )
                    x = jnp.einsum("bc,ce->be", batch, fc0)
                with jax.named_scope("layer1"):
                    x = legate.jax.with_sharding_constraint(
                        x, P("batch", "cxn")
                    )
                    batch = legate.jax.with_sharding_constraint(
                        batch1, P("batch", "cxn")
                    )
                    fc1 = legate.jax.with_sharding_constraint(
                        fc1, P("cxn", "ext")
                    )
                    scaled_batch = x * batch
                    return jnp.einsum("bc,ce->be", scaled_batch, fc1)

            return f(batch, params)

        batch_size = 8
        model_dim = 4
        with mesh, legate.jax.autoshard(True):
            f = jax.jit(
                c,
                in_shardings=(
                    (AUTO(mesh), AUTO(mesh)),
                    (AUTO(mesh), AUTO(mesh)),
                ),
                out_shardings=(AUTO(mesh)),
            )
            batch0 = jax.core.ShapedArray((batch_size, model_dim), np.float32)
            batch1 = jax.core.ShapedArray((batch_size, model_dim), np.float32)
            fc0 = jax.core.ShapedArray((model_dim, model_dim), np.float32)
            fc1 = jax.core.ShapedArray((model_dim, model_dim), np.float32)
            lowered = f.lower((batch0, batch1), (fc0, fc1)).compile()

            (batch0_sharding, batch1_sharding), (
                fc0_sharding,
                fc1_sharding,
            ) = lowered.input_shardings[0]

            self.assertTrue(isinstance(batch0_sharding, GSPMDSharding))
            self.assertEqual(
                batch0_sharding._hlo_sharding.tile_assignment_devices(),
                [0, 1, 2, 3],
            )
            self.assertTrue(
                batch0_sharding._hlo_sharding.tile_assignment_dimensions(),
                [4, 1],
            )
            self.assertEqual(
                batch1_sharding._hlo_sharding.tile_assignment_devices(), [0, 1]
            )
            self.assertTrue(
                batch1_sharding._hlo_sharding.tile_assignment_dimensions(),
                [2, 1],
            )

            replicated = NamedSharding(mesh, spec=P())
            self.assertEqual(fc0_sharding, replicated)

            self.assertTrue(isinstance(fc1_sharding, GSPMDSharding))

        def arg_maker():
            batch0 = make_shape(batch_size, model_dim)
            batch1 = make_shape(batch_size, model_dim)
            fc0 = make_shape(model_dim, model_dim)
            fc1 = make_shape(model_dim, model_dim)
            return (batch0, batch1), (fc0, fc1)

        full_mesh_sharding = PositionalSharding(jax.devices()[:4]).reshape(
            4, 1
        )

        legate_shardings = (
            (full_mesh_sharding, full_mesh_sharding),
            (None, fc1_sharding),
        )
        reference_shardings = (
            (full_mesh_sharding, full_mesh_sharding),
            (None, None),
        )

        with mesh, legate.jax.autoshard(True):
            self._test_against_reference(
                c, arg_maker, legate_shardings, reference_shardings
            )

    def test_reshard_explicitly_sharded_argument(self):
        if jax.device_count() != 4:
            self.skipTest("need 4 devices")

        mesh = Mesh(
            np.array(jax.devices()).reshape(4, 1, 1),
            ("batch", "cxn", "ext"),
        )

        logical_axes = [
            ("batch", "x"),
            ("embed", "x"),
            ("model", "y"),
        ]

        devices = np.array(jax.devices())

        def f(batch, param):
            param = legate.jax.with_sharding_constraint(param, P("cxn", "ext"))
            return jnp.einsum("bc,ce->be", batch, param)

        def c(batch, params):
            layer0 = legate.jax.task(
                f,
                name="layer0",
                devices=devices[:2].reshape(2, 1),
                device_axes=("x", "y"),
                logical_axes=logical_axes,
            )
            layer1 = legate.jax.task(
                f,
                name="layer1",
                devices=devices[2:].reshape(2, 1),
                device_axes=("x", "y"),
                logical_axes=logical_axes,
            )

            def inner(batch, params):
                (fc0, fc1) = params
                x = layer0(batch, fc0)
                return layer1(x, fc1)

            return inner(batch, params)

        batch_sharding = NamedSharding(mesh, P("batch", None))

        batch_size = 8
        model_dim = 4
        with mesh, legate.jax.autoshard(True):
            # use the legate jax mjit to annotate all arguments
            # with logical autosharding annotations
            jf = legate.jax.mjit(
                c,
                in_shardings=(
                    batch_sharding,
                    (AUTO(mesh), AUTO(mesh)),
                ),
                out_shardings=(AUTO(mesh)),
            )
            batch = jax.core.ShapedArray((batch_size, model_dim), np.float32)
            fc0 = jax.core.ShapedArray((model_dim, model_dim), np.float32)
            fc1 = jax.core.ShapedArray((model_dim, model_dim), np.float32)
            # make sure the complication succeeds
            _ = jf.lower(batch, (fc0, fc1)).compile()

        with mesh, legate.jax.autoshard(True):
            # use standard jit so that arguments are explicitly sharded
            # without logical autosharding annotations
            jf = jax.jit(
                c,
                in_shardings=(
                    batch_sharding,
                    (AUTO(mesh), AUTO(mesh)),
                ),
                out_shardings=(AUTO(mesh)),
            )
            batch = jax.core.ShapedArray((batch_size, model_dim), np.float32)
            fc0 = jax.core.ShapedArray((model_dim, model_dim), np.float32)
            fc1 = jax.core.ShapedArray((model_dim, model_dim), np.float32)
            # make sure the compilcation succeeds
            _ = jf.lower(batch, (fc0, fc1)).compile()

    def test_auto_shard_multiple_axes(self):
        if jax.device_count() < 8:
            self.skipTest("need >= 8 devices")

        logical_axes = [
            ("batch", "x"),
            ("embed", "x"),
            ("model", "y"),
        ]

        legate.jax.register_task(
            "embeddings",
            devices=[0, 1, 2, 3, 4, 5, 6, 7],
            dims=[4, 2],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        legate.jax.register_task(
            "layer0",
            devices=[0, 1, 2, 3],
            dims=[2, 2],
            device_axes=["x", "y"],
            logical_axes=logical_axes,
        )

        vocab = 2
        batch_size = 8
        seq = 4
        embed_size = 16

        def f(params, batch):
            embed, fc = params
            with jax.named_scope("embeddings"):
                batch = legate.jax.with_sharding_constraint(
                    batch, P(("batch", "model"), "seq")
                )
                embed = legate.jax.with_sharding_constraint(
                    embed, P("vocab", "embed")
                )
                batch = jax.nn.one_hot(batch, num_classes=vocab)
                x = jnp.einsum("bsv,ve->bse", batch, embed)
            with jax.named_scope("layer0"):
                x = legate.jax.with_sharding_constraint(
                    x, P(("batch", "model"), "seq", "embed")
                )
                fc = legate.jax.with_sharding_constraint(
                    fc, P("hidden", "embed")
                )
                x = jnp.einsum("bsh,eh->bsh", x, fc)
                x = legate.jax.with_sharding_constraint(
                    x, P("batch", "seq", "embed")
                )
                return x

        mesh = Mesh(
            np.array(jax.devices()).reshape(8, 1, 1),
            ("batch", "model", "embed"),
        )

        with mesh, legate.jax.autoshard(True):
            f = jax.jit(
                f,
                in_shardings=(AUTO(mesh), AUTO(mesh)),
                out_shardings=(AUTO(mesh)),
            )
            batch = jax.core.ShapedArray((batch_size, seq), np.float32)
            fc = jax.core.ShapedArray((embed_size, embed_size), np.float32)
            embed = jax.core.ShapedArray((vocab, embed_size), np.float32)
            params = (embed, fc)
            lowered = f.lower(params, batch).compile()

        _, fc_sharding = lowered.input_shardings[0][0]
        batch_sharding = lowered.input_shardings[0][1]
        output_sharding = lowered.output_shardings

        self.assertTrue(isinstance(fc_sharding, GSPMDSharding))
        self.assertEqual(
            fc_sharding._hlo_sharding.tile_assignment_devices(), [0, 1, 2, 3]
        )
        self.assertTrue(fc_sharding._hlo_sharding.replicate_on_last_tile_dim())

        self.assertTrue(isinstance(batch_sharding, GSPMDSharding))
        self.assertEqual(
            batch_sharding._hlo_sharding.tile_assignment_devices(),
            [0, 1, 2, 3, 4, 5, 6, 7],
        )
        self.assertFalse(
            batch_sharding._hlo_sharding.replicate_on_last_tile_dim()
        )

        self.assertTrue(isinstance(batch_sharding, GSPMDSharding))
        self.assertEqual(
            output_sharding._hlo_sharding.tile_assignment_devices(),
            [0, 1, 2, 3],
        )
        self.assertTrue(
            output_sharding._hlo_sharding.replicate_on_last_tile_dim()
        )

    def test_reshard_initial_final(self):
        self.skipTest(
            "do not support resharding across meshes with different sizes"
        )
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
            with mesh, legate.jax.autoshard(True):
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
        self.assertTrue(scale_sharding.is_equivalent_to(batch_sharding, 3))

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
            def f():
                batch = make_shape(batch_size, seq) % vocab
                fc0 = make_shape(embed_size, embed_size)
                fc1 = make_shape(embed_size, embed_size)
                scale = make_shape(batch_size, seq, embed_size)
                embed = make_shape(vocab, embed_size)
                params = (embed, fc0, fc1, scale)
                return (params, batch)

            return f()

        with mesh, legate.jax.autoshard(True):
            self._test_against_reference(
                c,
                arg_maker,
                arg_shardings=arg_shardings,
                reference_shardings=reference_shardings,
            )

    def test_mesh_wrapper(self):
        if jax.device_count() != 8:
            self.skipTest("need 8 devices")

        logical_axes = [
            ("batch", "x"),
            ("seq", "y"),
            ("embed", "z"),
        ]

        legate.jax.register_task(
            "layer0",
            devices=[0, 1, 2, 3],
            dims=[2, 1, 2],
            device_axes=["x", "y", "z"],
            logical_axes=logical_axes,
        )
        legate.jax.register_task(
            "layer1",
            devices=[4, 5, 6, 7],
            dims=[2, 1, 2],
            device_axes=["x", "y", "z"],
            logical_axes=logical_axes,
        )

        def partition(mesh, arg_shapes, result_shape):
            def lower_fn(x):
                return x

            out_sharding = NamedSharding(mesh, arg_shapes[0].sharding.spec)

            return (
                mesh,
                lower_fn,
                arg_shapes[0].sharding,
                (out_sharding,),
            )

        def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
            print(mesh)
            print(arg_shapes)
            print(result_shape)
            return arg_shapes[0].sharding

        def propagate_user_sharding(mesh, user_shape):
            return user_shape.sharding

        @custom_partitioning
        def layer(x):
            return x * x

        layer.def_partition(
            infer_sharding_from_operands=infer_sharding_from_operands,
            partition=partition,
            propagate_user_sharding=propagate_user_sharding,
        )

        def c(x):
            def f(x):
                with jax.named_scope("layer0"):
                    x = legate.jax.with_sharding_constraint(
                        x, P("batch", "seq", "embed")
                    )
                    x = layer(x)
                with jax.named_scope("layer1"):
                    x = legate.jax.with_sharding_constraint(
                        x, P("batch", "seq", "embed")
                    )
                    x = layer(x)
                return x

            return f(x)

        jax_mesh = Mesh(
            np.array(jax.devices()).reshape(4, 1, 2),
            ("batch", "seq", "embed"),
        )
        legate_mesh = MeshWrapper(jax_mesh, (2, 1, 2))

        batch = 8
        seq = 10
        embed = 8

        with legate_mesh, legate.jax.autoshard(True):
            c = jax.jit(
                c,
                in_shardings=(AUTO(legate_mesh),),
                out_shardings=(AUTO(legate_mesh)),
            )
            batch = jax.core.ShapedArray((batch, seq, embed), np.float32)
            with MeshWrapper.lower_mode():
                lowered = c.lower(batch)
            with MeshWrapper.compile_mode():
                compiled = lowered.compile()  # noqa: F841


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
