import jax
import jax._src.test_util as jtu

import legate.jax


class LegateJaxTestCase(jtu.JaxTestCase):
    def _test_against_cuda(self, f, arg_maker):
        with legate.jax.ignore_transforms():
            args = jax.jit(arg_maker, backend="cuda")()
            cuda_res = jax.jit(f, backend="cuda")(*args)

        # jax unfortunately caches the cuda jit compilation
        # before lowering, we need to clear it
        jax.clear_caches()
        args = jax.jit(arg_maker, backend="legate")()
        legate_res = jax.jit(f, backend="legate")(*args)
        self.assertAllClose(cuda_res, legate_res)
