from .legate_jax_impl import (
    no_op_custom_call,
    register_task,
    register_task_factory,
    shutdown,
    unregister_task,
    clear_tasks,
    compile_hlo_module,
)

from .legate_xla_compiler import register_custom_call_target

from .lib import (
    enable_fast_path,
    enable_tracing,
    enable_recomputation,
    init,
    ignore_transforms,
    with_sharding_constraint,
    ClientConfig,
    context,
)
from .task import (
    task,
    microbatch,
    parallelize,
    parallelize_step,
    shard_axes,
    Task,
)

from . import _version

__version__ = _version.get_versions()["version"]
