# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from jax_plugins.multimesh import init_test

init_test()

import jax
import jax._src.test_util as jtu
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import config
from jax.sharding import PositionalSharding

import multimesh.jax
from multimesh.jax.test_util import MultiMeshJaxTestCase

config.parse_flags_with_absl()


class TaskTest(MultiMeshJaxTestCase):
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
                res = multimesh.jax.task(f)(res, arrs)
            return res

        # allow the fast path to be chosen as a test that it is NOT taken
        # when a small module has MultiMesh custom calls
        self._test_against_reference(
            jnp_fxn, args_maker, enable_fast_path=True
        )

    def test_rematerialization(self):
        if jax.device_count() != 1:
            self.skipTest("need a single device")

        def c(x):
            def f(x):
                return x * x

            f = jax.checkpoint(
                f, policy=jax.checkpoint_policies.nothing_saveable
            )
            f = multimesh.jax.task(f)

            x = f(x)
            x = f(x)
            return x.sum()

        c = jax.value_and_grad(c)

        def arg_maker():
            return (self.make_shape(4, 4),)

        self._test_against_reference(c, arg_maker)

    def test_multiple_layer_rematerialization(self):
        if jax.device_count() != 1:
            self.skipTest("need a single device")

        def c(params, x):
            def f(params, x):
                y, z = params

                def layer(x, y, z):
                    return x * x + x * y + x * z

                layer = jax.checkpoint(
                    layer, policy=jax.checkpoint_policies.nothing_saveable
                )
                x = layer(x, y, z)
                x = layer(x, y, z)
                return layer(x, y, z)

            f = multimesh.jax.task(f)

            x = f(params, x)
            x = f(params, x)
            return x.sum()

        c = jax.value_and_grad(c)

        def arg_maker():
            x = self.make_shape(4, 4)
            y = self.make_shape(4, 4)
            z = self.make_shape(4, 4)
            return ((y, z), x)

        self._test_against_reference(c, arg_maker)

    def test_donation_single_argument(self):
        if jax.device_count() != 1:
            self.skipTest("need a single device")

        def c(x):
            def f(x):
                return x * x

            f = multimesh.jax.task(f)
            return f(x)

        def arg_maker():
            return (self.make_shape(4, 4),)

        self._test_against_reference(c, arg_maker, donate_argnums=(0,))

    def test_donation_multiple_arguments(self):
        self.skipTest("TODO: needs bug fix")

        def c(x, y, z):
            def f(x, y, z):
                return x * x, x * y, z

            f = multimesh.jax.task(f)

            def g(x, y, z):
                return x, x * y, z, x * z

            g = multimesh.jax.task(g)
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

            f = multimesh.jax.task(f)

            def g(x, y):
                return x * x, x * y

            g = multimesh.jax.task(g)
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

    def test_traced_tasks(self):
        if jax.device_count() != 2:
            self.skipTest("need 2 devices")
        self.skipTest("TODO: fix test")

        sharding = PositionalSharding(jax.devices()).reshape(2, 1)

        def step(params):
            def c(params):
                x, y = params

                def f(x):
                    return x * x

                f = multimesh.jax.task(f)

                def g(x, y):
                    return (x + (y * y)).sum()

                g = multimesh.jax.task(g)

                x = f(x)
                return g(x, y)

            c = jax.value_and_grad(c)
            loss, grads = c(params)
            params = jax.tree_map(
                lambda x, grad: x - 0.1 * grad, params, grads
            )
            return loss, params

        step = jax.jit(
            step,
            in_shardings=((sharding, sharding),),
            out_shardings=(None, (sharding, sharding)),
            donate_argnums=(0,),
        )

        def arg_maker():
            x = self.make_shape(4, 4)
            y = self.make_shape(4, 4)
            return x, y

        params = jax.jit(arg_maker, out_shardings=(sharding, sharding))()

        # run multiple iterations to cause tracing
        # iteration 0: establish stores
        # iteration 1: create trace
        # iteration 2: replay trace
        for i in range(3):
            expected_params = jax.tree_map(lambda x: x - 0.2 * x, params)
            x = params[0]
            expected_loss = 2 * (x * x).sum()
            loss, params = step(params)
            self.assertAllClose(expected_loss, loss)
            self.assertAllClose(expected_params, params)

    def test_task_gradients(self):
        if jax.device_count() != 1:
            self.skipTest("need a single device")

        def arg_maker():
            return jnp.arange(8, dtype=np.float32), 2

        def c(x, scale):
            def f(x, scale):
                return x * x * scale

            f = multimesh.jax.task(f)

            def g(x, y):
                return (jnp.cos(x) + y).sum()

            g = multimesh.jax.task(g)

            return g(f(x, scale), x)

        self._test_against_reference(jax.value_and_grad(c), arg_maker)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
