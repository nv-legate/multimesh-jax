# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from .legate_jax_impl import (
    no_op_custom_call,
    shutdown,
    replicate_parameters_smaller_than_num_elements,
    recompute_from_arguments_if_cost_less_than,
    clear_tasks,
    compile_hlo_module,
    set_startup_config,
)

from .legate_xla_compiler import register_custom_call_target

from .lib import (
    autoshard,
    context,
    enable_fast_path,
    enable_recomputation,
    enable_task_fusion,
    only_fuse_loop_tasks,
    ignore_transforms,
    with_sharding_constraint,
    tasks,
    mjit,
)

from .task import task, microbatch, parallelize, register_task

from .mesh import MeshWrapper

from . import _version

__version__ = _version.get_versions()["version"]
