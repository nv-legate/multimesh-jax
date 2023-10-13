# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        rng = jtu.rand_default(self.rng())

        def args_maker():
            return [rng(shape, dtype) for dtype in dtypes]

        def np_fxn(*arrs):
            res = arrs[:]
            for _ in range(3):
                res = [r + a for (r, a) in zip(res, arrs)]
            return res

        def task(lhs, rhs):
            return [l + r for (l, r) in zip(lhs, rhs)]

        task = legate.jax.task(task)

        def jnp_fxn(*arrs):
            res = arrs[:]
            for _ in range(3):
                res = task(res, arrs)
            return res

        self._CheckAgainstNumpy(np_fxn, jnp_fxn, args_maker)
        self._CompileAndCheck(jnp_fxn, args_maker)

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
