from .legate_jax_impl import (
    no_op_custom_call,
    shutdown,
    replicate_parameters_smaller_than_num_elements,
    set_store_cache_min_parallelism,
    fence,
    unregister_task,
    clear_tasks,
    compile_hlo_module,
)

from .legate_xla_compiler import register_custom_call_target

from .lib import (
    autoshard,
    context,
    enable_fast_path,
    enable_tracing,
    enable_recomputation,
    enable_task_fusion,
    only_fuse_loop_tasks,
    ignore_transforms,
    with_sharding_constraint,
    store_cache_min_parallelism,
    host_offload_min_reuse_distance,
    strict_static_order,
    ClientConfig,
    tasks,
    split_large_traces,
    mjit,
)

from .task import task, microbatch, parallelize, register_task

from . import _version

__version__ = _version.get_versions()["version"]
