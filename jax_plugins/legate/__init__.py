from .install_info import libpath
from pathlib import Path
import atexit


def initialize():
    import jax._src.xla_bridge as xb

    lib = Path(libpath) / "liblegate_plugin.so"
    c_api = xb.register_plugin(
        "legate", priority=500, library_path=str(lib), options=None
    )

    from jax.lib import xla_client
    from jaxlib import xla_extension as xe

    xla_client.register_custom_call_handler(
        "CUDA", xe.register_custom_call_target
    )

    import legate.jax

    atexit.register(legate.jax.shutdown)
