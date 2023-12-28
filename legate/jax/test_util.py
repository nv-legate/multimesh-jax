import jax
import jax._src.test_util as jtu

import legate.jax


class LegateJaxTestCase(jtu.JaxTestCase):
    def _test_against_native(self, f, arg_maker, arg_shardings=None):
        device_kind = jax.devices()[0].device_kind
        native_backend = "cpu" if device_kind == "cpu" else "cuda"
        jax.clear_caches()
        with legate.jax.ignore_transforms():
            if arg_shardings is not None:
                kwargs = dict(out_shardings=arg_shardings)
            else:
                kwargs = dict(backend=native_backend)
            args = jax.jit(arg_maker, **kwargs)()

            if arg_shardings is not None:
                kwargs = dict(in_shardings=arg_shardings)
            else:
                kwargs = dict(backend=native_backend)
            cuda_res = jax.jit(f, backend="cuda")(*args)

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
            for arg, arg_sharding, f_sharding in zip(
                args, arg_shardings, legate_f.input_shardings[0]
            ):
                self.assertTrue(
                    arg.sharding.is_equivalent_to(f_sharding, len(arg.shape))
                )
                self.assertTrue(
                    arg_sharding.is_equivalent_to(f_sharding, len(arg.shape))
                )
        legate_res = legate_f(*args)
        self.assertAllClose(cuda_res, legate_res)
