from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping, Optional, Sequence, Set, Tuple, Type

import gin
from tensorflow.compiler.xla.service import hlo_pb2


class LegateMatchers:
    key = re.compile("legate_key=(.*?)/")
    task_id = re.compile(r"task_id=(\d+)")


@dataclasses.dataclass(frozen=True)
class LegateKey:
    name: str
    is_backward: bool = False
    task_id: Optional[int] = None
    scopes: tuple[str] = dataclasses.field(default_factory=tuple)

    def __str__(self) -> str:
        name_str = self.name
        if self.is_backward:
            name_str += ".backward"
        if self.task_id is not None:
            name_str += f".{self.task_id}"
        for scope in self.scopes:
            name_str += f".{scope}"
        return name_str

    def __eq__(self, other) -> bool:
        return (
            self.name == other.name
            and self.is_backward == other.is_backward
            and self.task_id == other.task_id
        )

    def clone(
        self,
        *,
        include_task_id=True,
        include_backward=True,
        include_scope=True,
    ):
        return LegateKey(
            self.name,
            self.is_backward if include_backward else False,
            self.task_id if include_task_id else None,
            self.scopes if include_scope else (),
        )

    @staticmethod
    def set_task_id(instr: hlo_pb2.HloInstructionProto, task_id: int):
        repl = f"task_id={task_id}"
        new_metadata = LegateMatchers.task_id.sub(repl, instr.metadata.op_name)
        instr.metadata.op_name = new_metadata

    @staticmethod
    def metadata_string(
        name: str, task_id: Optional[int] = None, scopes: tuple[str] = ()
    ):
        keyvals = [f"name={name}"]
        if task_id is not None:
            keyvals.append(f"task_id={task_id}")
        if scopes:
            keyvals.append("scope={}".format(",".join(scopes)))
        return "legate_key={}/".format(";".join(keyvals))

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)

    def replace_key(self, metadata: str) -> str:
        def repl(match):
            return LegateKey.metadata_string(
                self.name, self.task_id, self.scopes
            )

        return LegateMatchers.key.sub(repl, metadata)

    def add_scope(self, scope: str) -> LegateKey:
        new_scopes = self.scopes + (scope,)
        return self.replace(scopes=new_scopes)

    @staticmethod
    def create(metadata: str) -> Optional[LegateKey]:
        match = LegateMatchers.key.search(metadata)
        name = None
        is_backward = False
        task_id = None
        scopes = ()
        if match:
            key_vals = match.groups()[0].split(";")
            for keyval in key_vals:
                key, val = keyval.split("=")
                if key == "name":
                    name = val
                elif key == "task_id":
                    task_id = int(val)
                elif key == "scope":
                    scopes = tuple(val.split(","))

            is_backward = (
                "legate_backprop" in metadata
                and "legate_forward" not in metadata
            )
            return LegateKey(name, is_backward, task_id, scopes)
        return None


def _find_unique_legate_key_in_subtree(
    instr: hlo_pb2.HloInstructionProto,
    *,
    computations: Mapping[int, hlo_pb2.HloComputationProto],
    keys: Optional[Set[str]] = None,
) -> Optional[LegateKey]:
    if keys is None:
        keys = set()
        _find_unique_legate_key_in_subtree(
            instr, computations=computations, keys=keys
        )
        if len(keys) > 1 or len(keys) == 0:
            return None
        return next(iter(keys))

    for comp_id in instr.called_computation_ids:
        comp = computations[comp_id]
        for subinstr in comp.instructions:
            key = LegateKey.create(subinstr.metadata.op_name)
            if key:
                keys.add(key)

            if len(keys) > 1:
                return None

            _find_unique_legate_key_in_subtree(
                subinstr, computations=computations, keys=keys
            )


class KeyTransform:
    def __call__(self, key: LegateKey) -> LegateKey:
        return key


@gin.configurable
class NameTransform(KeyTransform):
    def __init__(self, matcher: str, *group_transforms):
        self.matcher = re.compile(matcher)
        self.group_transforms = group_transforms

    def __call__(self, key: LegateKey) -> LegateKey:
        match = self.matcher.search(key.name)
        if match is None:
            return key

        new_groups = [
            (grp if tform is None else tform(grp))
            for tform, grp in zip(self.group_transforms, match.groups())
        ]
        new_name = "".join(map(str, new_groups))
        return LegateKey(
            name=new_name,
            is_backward=key.is_backward,
            task_id=key.task_id,
            scopes=key.scopes,
        )


@gin.configurable
class CombineNamesTransform(KeyTransform):
    def __init__(self, to_combine: Mapping[str, Sequence[str]]):
        self.to_combine = to_combine.copy()

    def __call__(self, key: LegateKey) -> LegateKey:
        for name, match_list in self.to_combine.items():
            for matcher in match_list:
                if matcher == key.name:
                    return LegateKey(
                        name=name,
                        is_backward=key.is_backward,
                        task_id=key.task_id,
                        scopes=key.scopes,
                    )
        # nothing matched, return the original key
        return key


@gin.configurable
class RemoveBackwardTransform(KeyTransform):
    def __init__(self, to_remove: Sequence[str]):
        self.to_remove = list(to_remove)

    def __call__(self, key: LegateKey) -> LegateKey:
        for name in self.to_remove:
            if key.is_backward and name == key.name:
                return LegateKey(
                    name=name,
                    is_backward=False,
                    task_id=key.task_id,
                    scopes=key.scopes,
                )
        # nothing matched, return the original key
        return key


@gin.configurable
@dataclasses.dataclass
class CompressLayersTransform(KeyTransform):
    matcher = r"(.*?)(\d+)(.*)"

    def __init__(
        self,
        num_layers: Optional[int] = None,
        num_procs: Optional[int] = None,
        num_interleaves: Optional[int] = None,
        num_layers_per_key: Optional[int] = None,
    ):
        if num_layers is not None and num_procs is not None:
            num_interleaves = 1 if num_interleaves is None else num_interleaves
            self.num_layers_per_key = (
                num_layers // num_procs // num_interleaves
            )
        elif num_layers_per_key is not None:
            self.num_layers_per_key = num_layers_per_key
        else:
            self.num_layers_per_key = 1

    def __call__(self, key: LegateKey) -> LegateKey:
        match = re.compile(self.matcher).search(key.name)
        if match:
            prefix, index, suffix = match.groups()
            new_index = int(index) // self.num_layers_per_key
            new_name = f"{prefix}{new_index}{suffix}"

            return key.replace(name=new_name)
        return key


@gin.configurable
def map_legate_key(
    key: LegateKey,
    transforms: Sequence[Tuple[Type[KeyTransform], Sequence[Any]]] = [],
):
    for transform_type, args in transforms:
        callable = transform_type(*args)
        key = callable(key)
    return key


def clear_legate_key(instr: hlo_pb2.HloInstructionProto):
    instr.metadata.op_name = LegateMatchers.key.sub("", instr.metadata.op_name)


def map_instruction_legate_key(instr: hlo_pb2.HloInstructionProto):
    key = LegateKey.create(instr.metadata.op_name)
    if key is None:
        return

    new_key = map_legate_key(key)
    if new_key != key:
        new_metadata = new_key.replace_key(instr.metadata.op_name)
        new_metadata += "/original_" + LegateKey.metadata_string(key)
        if key.is_backward and not new_key.is_backward:
            new_metadata = new_metadata.replace(
                "legate_backprop", "legate_forward"
            )
        instr.metadata.op_name = new_metadata
    return key
