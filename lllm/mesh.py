from __future__ import annotations

import abc
import re
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Type

import gin
import numpy as np
import numpy.typing as npt
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.service import hlo_pb2


class DeviceMesh(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def offset(self, device_offset: int) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def set_debug_mode() -> None:
        """Sets every mesh to use the processors 0..N so that
        they all run sequentially and use a small number of procs
        """
        raise NotImplementedError

    @property
    def tasks(self) -> List[TaskMesh]:
        raise NotImplementedError

    @property
    def ndevices(self) -> int:
        raise NotImplementedError


@gin.configurable
class TaskMesh(DeviceMesh):
    axes_regexp = re.compile("legate_axes=(.*?)/")

    def __init__(
        self,
        matcher: str,
        device_axes: Sequence[str],
        logical_axes: Sequence[Tuple[str, str]],
        device_shape: Optional[npt.ArrayLike] = None,
        device_offset: int = 0,
        devices: Optional[npt.ArrayLike] = None,
    ):
        self.matcher = re.compile(matcher)
        if devices is not None:
            self.devices = np.asarray(devices)
        elif device_shape is not None:
            device_shape = np.asarray(device_shape)
            self.devices = np.arange(device_shape.prod()).reshape(device_shape)
            self.devices += device_offset
        else:
            raise Exception(
                "one of devices or device_shape must not be None in TaskMesh"
            )
        self.device_axes: Mapping[str, int] = dict(
            (name, ax) for ax, name in enumerate(device_axes)
        )
        self.logical_axes = logical_axes

        self._min_device = np.min(self.devices)
        self._max_device = np.max(self.devices)

    def __hash__(self):
        return (
            hash(self.devices.data.tobytes())
            ^ hash(frozenset(self.device_axes.items()))
            ^ hash(frozenset(self.logical_axes))
        )

    def __eq__(self, other):
        return (
            self.devices.shape == other.devices.shape
            and self.device_axes == other.device_axes
            and self.logical_axes == other.logical_axes
        )

    def __str__(self):
        dev_axes = [(id, name) for (name, id) in self.device_axes.items()]
        dev_axes.sort()
        dev_axes_str = " ".join([f"{id}:{name}" for id, name in dev_axes])
        dev_str = str(self.devices).replace("\n", "")
        return "Mesh {}: devices={} device_axes=[{}] logical_axes={}".format(
            self.matcher.pattern, dev_str, dev_axes_str, self.logical_axes
        )

    def matches(self, name: str) -> bool:
        return bool(self.matcher.fullmatch(name))

    def partially_matches(self, name: str) -> bool:
        return bool(self.matcher.search(name))

    def get_device_range(self) -> slice:
        return slice(self._min_device, self._max_device + 1)

    @classmethod
    def create(cls, **kwargs) -> DeviceMesh:
        return TaskMesh(**kwargs)

    def offset(self, device_offset: int) -> None:
        self.devices += device_offset
        self._min_device += device_offset
        self._max_device += device_offset

    def set_debug_mode(self) -> None:
        # set the devices to start at 0
        offset = -self._min_device
        self.offset(offset)

    @property
    def tasks(self) -> List[TaskMesh]:
        return [self]

    @property
    def ndevices(self) -> int:
        return self.devices.size

    @classmethod
    def get_legate_axes(
        self, instr: hlo_pb2.HloInstructionProto
    ) -> Tuple[str, ...]:
        match = self.axes_regexp.search(instr.metadata.op_name)
        if match:
            try:
                # This should be a tuple string like ('x', 'y')
                axes = eval(match.groups()[0])
                # TODO, figure out why this is creating nested tuples sometimes
                if axes and isinstance(axes[0], tuple):
                    return axes[0]
                return axes
            except SyntaxError:
                raise Exception(
                    "Bad axis definition for {}: {}".format(
                        instr.name, instr.metadata.op_name
                    )
                )
        else:
            return None

    def shard(self, instr: hlo_pb2.HloInstructionProto) -> None:
        """Convert op metadata to op sharding (if such metadata exists)
           and add the op_sharding to the input instruction.

        Args:
            instr (hlo_pb2.HloInstructionProto): The instruction to add
            sharding annotations to (in-place)
        """

        axes = self.get_legate_axes(instr)
        if axes is None:
            return

        sharding = xla_data_pb2.OpSharding()
        sharding.type = xla_data_pb2.OpSharding.Type.OTHER
        sharding.tile_shape.CopyFrom(instr.shape)
        total_size = 1

        # The assignment of device axes to logical axes is based on a
        # prioritized list. Mappings are given as a sequence of tuples
        # with (device, logical), e.g.
        # ("x", "batch")
        # ("y", "data")
        # ("z", "data")
        # ("x", "data")
        # This says to first assign the batch device axis to logical axis y
        # If the tensor as logical axis y or z, assigned to device axis data
        # Finally, if the data device axis is not used (no y,z in tensor),
        # partition the x-axis on both batch and data device axes.

        # if a logical axis is repeated, only use the outermost axis
        # for sharding. We do not want to double-shard an instruction
        # if e.g. the x logical axis appears twice
        # if a logical axis is repeated, but has multiple device axes
        # matching the logical axis assign them in order
        # e.g. if a tensor is ('x', 'x', 'y') and 'x' matches
        # both 'data' and 'batch', then the tensor can be sharded as
        # ('data', 'batch', None)

        axes = self.get_legate_axes(instr)

        # Multiple device axes can be assigned to a logical axis
        logical_axes = dict((ax, []) for ax in axes)
        # A device axis can only be assigned to a single logical axis
        device_axis_assignment = {}
        for logical_ax, device_ax in self.logical_axes:
            if logical_ax in axes and device_ax not in device_axis_assignment:
                device_axis_assignment[device_ax] = logical_ax
                # But a logical axis can be assigned to multiple devices axes
                logical_axes[logical_ax].append(device_ax)

        # if none of the device axes got matched, this is not sharded
        if not device_axis_assignment:
            return

        # Group together the indices of all axes that have same name
        logical_axis_numbers = dict((ax, []) for ax in axes)
        for idx, ax in enumerate(axes):
            if ax is not None:
                logical_axis_numbers[ax].append(idx)

        instr.metadata.op_name += f"/mesh_axes={logical_axes}/"

        tile_dims = [1] * len(instr.shape.dimensions)
        for logical_ax, axis_nums in logical_axis_numbers.items():
            device_axes = logical_axes[logical_ax]
            if len(device_axes) == len(axis_nums):
                # either a 1-1 match between logical/device axis
                # or a logical axis is repeated and there is
                # the same number of matching device axes
                for device_ax, axis in zip(device_axes, axis_nums):
                    tile_dims[axis] = self.devices.shape[
                        self.device_axes[device_ax]
                    ]
            elif len(device_axes) == 1:
                # a logical axis name is used more than once
                # and there is exactly one device axis that matches
                # that logical name
                device_ax = device_axes[0]
                axis = axis_nums[-1]  # use the outermost match
                tile_dims[axis] = self.devices.shape[
                    self.device_axes[device_ax]
                ]
            elif len(axis_nums) == 1:
                # more than one device axis assigned to a single logical axis
                axis = axis_nums[0]
                tile_dim = 1
                for device_ax in device_axes:
                    tile_dim *= self.devices.shape[self.device_axes[device_ax]]
                tile_dims[axis] = tile_dim
            elif len(device_axes) == 0:
                # no sharding on these dims
                for axis in axis_nums:
                    tile_dims[axis] = 1
            else:
                raise Exception(
                    f"mismatched logical and device axes with multiple "
                    f"matches for axes={axes} on {instr.name}: "
                    f"{device_axes} assigned to {axis_nums}"
                )

        total_size = 1
        for dim in tile_dims:
            sharding.tile_assignment_dimensions.append(dim)
            total_size *= dim

        if total_size < self.devices.size:
            extra_dim = self.devices.size // total_size
            sharding.tile_assignment_dimensions.append(extra_dim)
            sharding.replicate_on_last_tile_dim = True
            raise Exception(
                f"{instr.name} with shape={instr.shape.dimensions} "
                f"has partial replication on axes={axes}"
            )
        # TODO: support a different device ordering
        # Devices should be relative numberings from 0...n
        sharding.tile_assignment_devices.extend(np.arange(self.devices.size))
        instr.sharding.CopyFrom(sharding)
        dim_prod = 1
        for entry in sharding.tile_assignment_dimensions:
            dim_prod *= entry

        if dim_prod != len(sharding.tile_assignment_devices):
            raise Exception(
                f"no. devices={len(sharding.tile_assignment_devices)} does "
                f" not match product of "
                f"tiling={sharding.tile_assignment_dimensions} for "
                f"axes={axes} shape={instr.shape.dimensions}"
            )


@gin.configurable
class LayerMesh(DeviceMesh):
    """A mesh for a sequence of layers all with the same name.
    For name = 'encoder_{}', this would produce a sequence of TaskMesh
    objects with names encoder_0, encoder_1, ..., each assigned the same
    number of devices.
    """

    def __init__(
        self,
        matcher_template: str,
        num_layers: int,
        device_axes: Sequence[str],
        logical_axes: Mapping[str, str],
        device_shape: npt.ArrayLike,
        layer_offset: int = 0,
    ):
        self.task_meshes: List[TaskMesh] = []
        self.device_shape = np.asarray(device_shape)
        device_size = self.device_shape.prod()
        device_offset = 0

        for lyr in range(num_layers):
            task_matcher = matcher_template.format(layer_offset + lyr)
            task_mesh = TaskMesh(
                task_matcher,
                device_axes,
                logical_axes,
                device_shape=device_shape,
            )
            task_mesh.offset(device_offset)

            self.task_meshes.append(task_mesh)
            device_offset += device_size

    @classmethod
    def create(cls, **kwargs) -> DeviceMesh:
        return LayerMesh(**kwargs)

    @property
    def ndevices(self) -> int:
        return self.device_shape.prod() * len(self.tasks)

    @property
    def tasks(self) -> List[TaskMesh]:
        return self.task_meshes

    def set_debug_mode(self) -> None:
        for mesh in self.task_meshes:
            mesh.set_debug_mode()

    def offset(self, device_offset: int) -> None:
        for mesh in self.task_meshes:
            mesh.offset(device_offset)

    def layer_name(self, layer: int) -> str:
        if self.num_layers is None:
            return self.name
        else:
            self.canonical_name(self.name, layer)

    @staticmethod
    def canonical_name(name: str, layer: int):
        return f"{name}_{layer}"


# Gin doesn't play nice with class inheritance so we have
# to create an entirely independent class
@gin.configurable
class LastLayerMesh:
    @classmethod
    def create(self, *, num_layers: int, **kwargs) -> DeviceMesh:
        mesh = TaskMesh(**kwargs)
        offset = (num_layers - 1) * mesh.ndevices
        mesh.offset(offset)
        return mesh


@gin.configurable
class InterleavedLayerMesh(DeviceMesh):
    def __init__(
        self,
        matcher_template: str,
        num_layers: int,
        num_interleaves: int,
        device_axes: Sequence[str],
        logical_axes: Mapping[str, str],
        device_shape: npt.ArrayLike,
    ):
        num_layers_per_interleave = num_layers // num_interleaves
        if num_layers % num_interleaves:
            raise Exception(
                f"number_of layers ({num_layers} not evenly divided by"
                f" number of interleaves {num_interleaves}"
            )

        self.layers = [
            LayerMesh(
                matcher_template,
                num_layers_per_interleave,
                device_axes,
                logical_axes,
                device_shape,
                i * num_layers_per_interleave,
            )
            for i in range(num_interleaves)
        ]

        self._ndevices = self.layers[0].ndevices * num_interleaves

    @classmethod
    def create(self, **kwargs) -> DeviceMesh:
        return InterleavedLayerMesh(**kwargs)

    @property
    def ndevices(self) -> int:
        return self._ndevices

    @property
    def tasks(self) -> List[TaskMesh]:
        return [mesh for layer in self.layers for mesh in layer.tasks]

    def set_debug_mode(self) -> None:
        for layer in self.layers:
            layer.set_debug_mode()

    def offset(self, device_offset: int) -> None:
        for layer in self.layers:
            layer.offset(device_offset)


@gin.configurable
class PipelineMesh(DeviceMesh):
    def __init__(self, meshes: Sequence[DeviceMesh]):
        self.meshes = meshes
        device_offset = 0
        for mesh in meshes:
            mesh.offset(device_offset)
            device_offset += mesh.ndevices
        self.nprod = device_offset

    @classmethod
    def create(
        self, meshes: Sequence[Tuple[Type[DeviceMesh], Mapping[str, Any]]]
    ) -> DeviceMesh:
        meshes = [cls.create(**cls_args) for cls, cls_args in meshes]
        return PipelineMesh(meshes)

    def set_debug_mode(self) -> None:
        for mesh in self.meshes:
            mesh.set_debug_mode()

    def offset(self, device_offset: int) -> int:
        for mesh in self.meshes:
            mesh.offset(device_offset)

    @property
    def ndevices(self) -> int:
        return self.nproc

    @property
    def tasks(self) -> List[TaskMesh]:
        task_meshes = []
        for mesh in self.meshes:
            task_meshes.extend(mesh.tasks)
        return task_meshes


@gin.configurable
class GlobalMesh:
    def __init__(self, meshes: Sequence[DeviceMesh] = [], debug_mode=False):
        if debug_mode:
            for mesh in meshes:
                mesh.set_debug_mode()

        self.tasks: List[TaskMesh] = [
            task for mesh in meshes for task in mesh.tasks
        ]

    def __str__(self):
        str_arr = ["Global Mesh"]
        meshes = [str(task_mesh) for task_mesh in self.tasks]
        if meshes:
            str_arr.extend(meshes)
        else:
            str_arr.append(" empty")
        return "\n".join(str_arr)

    def get_task_mesh(self, name: str) -> Optional[TaskMesh]:
        for task in self.tasks:
            if task.matches(name):
                return task
        for task in self.tasks:
            if task.partially_matches(name):
                return task
        return None

    def add_task_mesh(self, task_mesh: TaskMesh) -> None:
        self.tasks[task_mesh.name] = task_mesh


@gin.configurable
def Mesh(
    mesh_type: Type[DeviceMesh], args: Tuple[Mapping[str, Any], ...]
) -> DeviceMesh:
    kwargs = {}
    for arg in args:
        kwargs.update(arg)
    return mesh_type(**kwargs)


@gin.configurable()
def legate_global_mesh(
    mesh_args: Sequence[
        Tuple[Type[DeviceMesh], str, Tuple[Mapping[str, Any], ...]]
    ] = [],
    debug_mode: bool = False,
) -> Optional[GlobalMesh]:
    if not mesh_args:
        return None

    meshes = []
    for typ, matcher, args in mesh_args:
        kwargs = {}
        for arg in args:
            kwargs.update(arg)
        meshes.append(typ(matcher, **kwargs))

    return GlobalMesh(meshes, debug_mode)
