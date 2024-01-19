import jax
import jax._src.test_util as jtu
from jax.tree_util import tree_map

import legate.jax


class LegateJaxTestCase(jtu.JaxTestCase):
    def _test_against_reference(
        self, f, arg_maker, arg_shardings=None, reference_shardings=None
    ):
        device_kind = jax.devices()[0].device_kind
        reference_backend = "cpu" if device_kind == "cpu" else "cuda"
        jax.clear_caches()

        if reference_shardings is None:
            reference_shardings = arg_shardings

        with legate.jax.ignore_transforms():
            if reference_shardings is not None:
                kwargs = dict(out_shardings=reference_shardings)
            else:
                kwargs = dict(backend=reference_backend)

            args = jax.jit(arg_maker, **kwargs)()
            if reference_shardings is not None:
                kwargs = dict(in_shardings=reference_shardings)
            else:
                kwargs = dict(backend=reference_backend)
            cuda_res = jax.jit(f, **kwargs)(*args)

        # jax unfortunately caches the cuda jit compilation
        # before lowering, we need to clear it
        jax.clear_caches()
        if arg_shardings is not None:
            kwargs = dict(out_shardings=arg_shardings)
        else:
            kwargs = dict(backend="legate")

        args = jax.jit(arg_maker, **kwargs)()
        if arg_shardings is not None:
            kwargs = dict(in_shardings=arg_shardings)
        else:
            kwargs = dict(backend="legate")

        legate_f = jax.jit(f, **kwargs).lower(*args).compile()
        if arg_shardings is not None:

            def compare_assert(arg, arg_sharding, f_sharding):
                if arg_sharding is None:
                    self.assertTrue(f_sharding.is_fully_replicated)
                else:
                    self.assertTrue(
                        arg.sharding.is_equivalent_to(
                            f_sharding, len(arg.shape)
                        )
                    )
                    self.assertTrue(
                        arg_sharding.is_equivalent_to(
                            f_sharding, len(arg.shape)
                        )
                    )

            tree_map(
                compare_assert,
                args,
                arg_shardings,
                legate_f.input_shardings[0],
            )

        legate_res = legate_f(*args)
        self.assertAllClose(cuda_res, legate_res)
