# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Sequence

import jax
import jax._src.test_util as jtu
from jax.tree_util import tree_map

import multimesh.jax


@jtu.with_config(jax_numpy_rank_promotion="allow")
class MultiMeshJaxTestCase(jtu.JaxTestCase):
    def _test_against_reference(
        self,
        f,
        arg_maker,
        arg_shardings: Optional[Sequence[Any]] = None,
        reference_shardings: Optional[Sequence[Any]] = None,
        donate_argnums: Optional[Sequence[int]] = None,
        enable_fast_path: bool = False,
        atol=None,
        rtol=None,
    ):
        reference_backend = "cpu"
        jax.clear_caches()

        if reference_shardings is None:
            reference_shardings = arg_shardings

        with multimesh.jax.ignore_transforms():
            if reference_shardings is not None:
                kwargs = dict(out_shardings=reference_shardings)
            else:
                kwargs = dict(backend=reference_backend)

            args = jax.jit(arg_maker, **kwargs)()

            if reference_shardings is not None:
                kwargs = dict(in_shardings=reference_shardings)
            else:
                kwargs = dict(backend=reference_backend)

            cuda_res = jax.jit(f, donate_argnums=donate_argnums, **kwargs)(
                *args
            )

        # jax unfortunately caches the cuda jit compilation
        # before lowering, we need to clear it
        jax.clear_caches()
        if arg_shardings is not None:
            kwargs = dict(out_shardings=arg_shardings)
        else:
            kwargs = dict(backend="multimesh")

        with multimesh.jax.enable_fast_path(enable_fast_path):
            args = jax.jit(arg_maker, **kwargs)()
            if arg_shardings is not None:
                kwargs = dict(in_shardings=arg_shardings)
            else:
                kwargs = dict(backend="multimesh")

            mm_f = (
                jax.jit(f, donate_argnums=donate_argnums, **kwargs)
                .lower(*args)
                .compile()
            )
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
                    mm_f.input_shardings[0],
                )

            mm_res = mm_f(*args)
        self.assertAllClose(cuda_res, mm_res, atol=atol, rtol=rtol)
