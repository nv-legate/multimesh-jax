import contextlib
import enum
import functools
from collections import OrderedDict

import numpy as np
from jax._src import config as jax_config
from jax._src.mesh import thread_resources
from jax.sharding import AbstractMesh


class MeshWrapper(contextlib.ContextDecorator):
    class Mode(enum.IntEnum):
        LOWERING = 0
        COMPILING = 1
        NONE = 2

    _mode = Mode.NONE

    def __init__(self, global_mesh, local_mesh: tuple[int]):
        self.global_mesh = global_mesh
        self.local_shape = OrderedDict(
            (name, dim) for name, dim in self.global_mesh.shape.items()
        )
        for local_dim, (name, global_dim) in zip(
            local_mesh, self.global_mesh.shape.items()
        ):
            if local_dim != global_dim:
                self.local_shape[name] = local_dim

        self.num_local_devices = np.prod(local_mesh)

        local_devices_slice = tuple(slice(0, dim) for dim in local_mesh)
        self._local_devices = self.global_mesh.devices[local_devices_slice]

    @property
    def shape_tuple(self):
        return tuple(
            (name, size)
            for name, size in zip(self.axis_names, self.devices.shape)
        )

    @functools.cached_property
    def abstract_mesh(self):
        return AbstractMesh(self.shape_tuple, axis_types=self.axis_types)

    @classmethod
    @contextlib.contextmanager
    def mode(cls, mode: Mode):
        save = cls._mode
        cls._mode = mode
        yield
        cls._mode = save

    @classmethod
    @contextlib.contextmanager
    def compile_mode(cls):
        with cls.mode(MeshWrapper.Mode.COMPILING):
            yield

    @classmethod
    @contextlib.contextmanager
    def lower_mode(cls):
        with cls.mode(MeshWrapper.Mode.LOWERING):
            yield

    def __repr__(self):
        return f"MeshWrapper({repr(self.global_mesh)})"

    def __str__(self):
        return f"MeshWrapper({str(self.global_mesh)})"

    @property
    def is_multi_process(self):
        return self.global_mesh.is_multi_process

    @property
    def _flat_devices_tuple(self):
        devices = self.global_mesh._flat_devices_tuple
        if self.compiling():
            return devices[: self.num_local_devices]
        return devices

    @property
    def local_devices(self):
        return self.global_mesh.local_devices

    def compiling(self) -> bool:
        return MeshWrapper._mode == self.Mode.COMPILING

    @property
    def axis_names(self):
        return self.global_mesh.axis_names

    @functools.cached_property
    def _name_to_type(self):
        return self.global_mesh._name_to_type

    @functools.cached_property
    def axis_types(self):
        return self.global_mesh.axis_types

    @property
    def axis_sizes(self) -> tuple[int, ...]:
        if self.compiling():
            return tuple(self.local_shape.values())
        return self.global_mesh.axis_sizes

    @property
    def devices(self):
        if self.compiling():
            raise Exception("local!")
            return self._local_devices
        return self.global_mesh.devices

    @property
    def shape(self):
        if self.compiling():
            return self.local_shape
        return self.global_mesh.shape

    @property
    def _internal_device_list(self):
        return self.global_mesh._internal_device_list

    @property
    def size(self):
        if self.compiling():
            return self.num_local_devices
        return self.global_mesh.size

    @property
    def empty(self):
        return self.global_mesh.empty

    def __enter__(self):
        new_env = thread_resources.stack[-1].with_mesh(self)
        thread_resources.stack.append(new_env)
        thread_resources.env = new_env
        jax_config.mesh_context_manager.set_local(
            tuple(
                t.physical_mesh
                for t in thread_resources.stack
                if not t.physical_mesh.empty
            )
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        thread_resources.stack.pop()
        thread_resources.env = thread_resources.stack[-1]
        jax_config.mesh_context_manager.set_local(
            tuple(
                t.physical_mesh
                for t in thread_resources.stack
                if not t.physical_mesh.empty
            )
        )
        return False
