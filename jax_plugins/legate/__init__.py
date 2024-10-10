from .install_info import libpath
from pathlib import Path
import atexit
import functools

from .init import init, init_test


def initialize():
    import jax._src.xla_bridge as xb
    from legate.jax import register_custom_call_target

    lib = Path(libpath) / "liblegate_plugin.so"
    c_api = xb.register_plugin(
        "legate", priority=500, library_path=str(lib), options=None
    )

    from jax._src.lib import xla_client
    from jaxlib import xla_extension as xe

    xla_client.register_custom_call_handler(
        "CUDA", functools.partial(register_custom_call_target, c_api)
    )

    import legate.jax

    atexit.register(legate.jax.shutdown)
