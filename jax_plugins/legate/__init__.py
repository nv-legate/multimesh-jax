from .install_info import libpath
from pathlib import Path


def initialize():
    import jax._src.xla_bridge as xb

    lib = Path(libpath) / "liblegate_plugin.so"
    xb.register_plugin(
        "legate", priority=500, library_path=str(lib), options=None
    )
