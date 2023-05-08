# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any, List, Mapping, Optional, Sequence, Set, Tuple, Union

import gin
import numpy as np
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.service import hlo_pb2

import legate.core.types as ty
from legate.core import (
    Library,
    Machine,
    Rect,
    Store,
    get_legate_runtime,
    get_machine,
)
from legate.core.shape import Shape
from legate.core.store import StorePartition
from legate.core.types import ReductionOp
from legate.core.utils import OrderedSet

from .hlo_utils import (
    _SIZE_PRESERVING_OPS,
    compute_sharding_propagation,
    find_entry_computation,
    find_root_instruction,
    is_replicated_instruction,
    is_scalar,
    is_tuple_shape,
)
from .key_utils import LegateKey, clear_legate_key, map_instruction_legate_key
from .mesh import GlobalMesh, TaskMesh

_CODES_TO_DTYPES = {
    xla_data_pb2.PrimitiveType.PRED: ty.bool_,
    xla_data_pb2.PrimitiveType.S8: ty.int8,
    xla_data_pb2.PrimitiveType.S16: ty.int16,
    xla_data_pb2.PrimitiveType.S32: ty.int32,
    xla_data_pb2.PrimitiveType.S64: ty.int64,
    xla_data_pb2.PrimitiveType.U8: ty.uint8,
    xla_data_pb2.PrimitiveType.U16: ty.uint16,
    xla_data_pb2.PrimitiveType.U32: ty.uint32,
    xla_data_pb2.PrimitiveType.U64: ty.uint64,
    xla_data_pb2.PrimitiveType.F16: ty.float16,
    xla_data_pb2.PrimitiveType.BF16: ty.float16,
    xla_data_pb2.PrimitiveType.F32: ty.float32,
    xla_data_pb2.PrimitiveType.F64: ty.float64,
}

_CHEAP_REPLICATED_OPS = {"reshape", "convert", "broadcast", "constant"}


@gin.configurable
@dataclass
class RunConfig:
    explicit_replication: bool = False
    rematerialization: bool = False


class LLMLib(Library):
    def __init__(self, name: str) -> None:
        self.name = name
        self.runtime: Union[LLMRuntime, None] = None
        self.shared_object: Any = None

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from lllm.install_info import libpath

        return os.path.join(
            libpath, f"liblegate_xla{self.get_library_extension()}"
        )

    def get_c_header(self) -> str:
        from lllm.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "legate_xla_perform_registration"

    def initialize(self, shared_object: Any) -> None:
        assert self.runtime is None
        self.shared_object = shared_object

    def set_runtime(self, runtime: LLMRuntime) -> None:
        assert self.runtime is None
        assert self.shared_object is not None
        self.runtime = runtime

    def destroy(self) -> None:
        if self.runtime is not None:
            self.runtime.destroy()


llm_lib = LLMLib("legate.xla")
llm_context = get_legate_runtime().register_library(llm_lib)
_llm = llm_lib.shared_object


@unique
class LLMOpCode(IntEnum):
    HLO_LOADER = _llm.HLO_PROTOTYPE_LOAD
    HLO_EXECUTOR = _llm.HLO_PROTOTYPE_EXECUTE
    HLO_FILL = _llm.HLO_PROTOTYPE_FILL
    DISTRIBUTED_INIT = _llm.HLO_PROTOTYPE_DISTRIBUTED_INIT
    DISTRIBUTED_SHUTDOWN = _llm.HLO_PROTOTYPE_DISTRIBUTED_SHUTDOWN


# @unique
# class LLMTunable(IntEnum):
#    NUM_GPUS = _llm.LLM_TUNABLE_NUM_GPUS
#    NUM_PROCS = _llm.LLM_TUNABLE_NUM_PROCS


class LLMRuntime:
    def __init__(self, legate_context):
        self.legate_context = legate_context
        self.legate_runtime = get_legate_runtime()
        self.run_id = 0
        llm_lib.set_runtime(self)
        assert llm_lib.shared_object is not None

        self._machine = get_machine()
        self._machine = self._machine.only(self._machine.preferred_kind)
        self._launch_domain = Rect([self._machine.num_procs])
        self._next_hlo_id = 100
        self._layer_name_to_hlo_id: dict[str, int] = {}

    def destroy(self) -> None:
        pass

    def issue_execution_fence(self, block: bool = False) -> None:
        self.legate_runtime.issue_execution_fence(block=block)

    def fill_store_partitions(
        self, machine: Machine, stores: List[StorePartition]
    ) -> None:
        with machine:
            launch_domain = Rect(
                lo=[0], hi=[machine.num_procs], exclusive=True
            )
            task = self.legate_context.create_manual_task(
                LLMOpCode.HLO_FILL,
                launch_domain=launch_domain,
            )
            for store in stores:
                task.add_output(store)
            task.execute()

    def fill_stores(self, machine: Machine, stores: List[Store]) -> None:
        with machine:
            task = self.legate_context.create_auto_task(
                LLMOpCode.HLO_FILL,
            )
            for store in stores:
                task.add_output(store)
            task.execute()

    def init_distributed(
        self,
        coordinator_addr: Optional[str] = None,
        coordinator_port: int = 1234,
    ):
        machine = self._machine
        launch_domain = Rect(lo=[0], hi=[machine.num_procs], exclusive=True)
        runtime.legate_runtime.set_provenance("init_distributed")
        with machine:
            task = self.legate_context.create_manual_task(
                LLMOpCode.DISTRIBUTED_INIT, launch_domain=launch_domain
            )
            task.add_scalar_arg(coordinator_addr, ty.string)
            task.add_scalar_arg(coordinator_port, ty.int32)
            task.set_side_effect(True)
            # Workers will block waiting for results from other
            # workers in the cooordination service.
            task.set_concurrent(True)
            task.execute()
            runtime.legate_runtime.reset_provenance()
            # all tasks must wait for this to finish
            runtime.issue_execution_fence()

    def _next_run_id(self) -> int:
        id = self.run_id
        self.run_id += 1
        return id

    def shutdown_distributed(self):
        machine = self._machine
        launch_domain = Rect(lo=[0], hi=[machine.num_procs], exclusive=True)
        runtime.issue_execution_fence()
        with machine:
            runtime.legate_runtime.set_provenance("shutdown_distributed")
            task = self.legate_context.create_manual_task(
                LLMOpCode.DISTRIBUTED_SHUTDOWN, launch_domain=launch_domain
            )
            task.set_side_effect(True)
            # Workers will block waiting for results from other
            # workers in the cooordination service.
            task.set_concurrent(True)
            task.execute()
            runtime.legate_runtime.reset_provenance()

    def load_hlo(
        self,
        hlo_string: str,
        hlo_name: str,
        mesh: Optional[TaskMesh] = None,
        debug: bool = False,
    ) -> int:
        if mesh is None:
            machine = self._machine
            nproc = machine.num_procs
        elif debug:
            machine = self._machine
            nproc = (
                mesh.get_device_range().stop - mesh.get_device_range().start
            )
        else:
            sl = mesh.get_device_range()
            if sl.stop > self._machine.num_procs:
                raise ValueError(
                    "Bad processor range for running this module; the module "
                    f"requested processors [{sl.start}...{sl.stop}), but the "
                    f"runtime has only {self._machine.num_procs} processors."
                )
            machine = self._machine[sl]
            nproc = machine.num_procs

        launch_domain = Rect(lo=[0], hi=[machine.num_procs], exclusive=True)
        with machine:
            runtime.legate_runtime.set_provenance(hlo_name)
            task = self.legate_context.create_manual_task(
                LLMOpCode.HLO_LOADER,
                launch_domain=launch_domain,
            )
            hlo_id = self._next_hlo_id
            self._next_hlo_id += 1
            task.add_scalar_arg(self._next_run_id(), ty.uint64)
            task.add_scalar_arg(hlo_string, ty.string)
            task.add_scalar_arg(hlo_name, ty.string)
            task.add_scalar_arg(hlo_id, ty.uint64)
            task.add_scalar_arg(nproc, ty.uint64)
            task.set_side_effect(True)
            task.set_concurrent(True)
            task.execute()
            runtime.legate_runtime.reset_provenance()

            return hlo_id

    def execute_hlo(
        self,
        hlo_id: int,
        name: str,
        inputs: List[Union[Store, StorePartition]],
        outputs: List[Union[Store, StorePartition]],
        red_ops: List[Optional[IntEnum]],
        mesh: Optional[TaskMesh] = None,
        init_tensors: bool = False,
    ) -> None:
        if mesh is None:
            machine = self._machine
        else:
            sl = mesh.get_device_range()
            machine = self._machine[sl]

        launch_domain = Rect(lo=[0], hi=[machine.num_procs], exclusive=True)

        for red_op, output in zip(red_ops, outputs):
            if red_op is None:
                if (
                    not isinstance(output, StorePartition)
                    and machine.num_procs > 1
                ):
                    raise Exception(
                        f"received non-partitioned store as output: {output}"
                    )

        if init_tensors:
            partitions = [
                store for store in inputs if isinstance(store, StorePartition)
            ]
            stores = [store for store in inputs if isinstance(store, Store)]
            reductions = [
                store
                for (op, store) in zip(red_ops, outputs)
                if op is not None
            ]
            runtime.legate_runtime.push_provenance(
                f"fill partitioned inputs: {name}"
            )
            self.fill_store_partitions(machine, partitions)
            runtime.legate_runtime.pop_provenance()
            runtime.legate_runtime.push_provenance(
                f"fill unpartitioned inputs: {name}"
            )
            self.fill_stores(machine, stores + reductions)
            runtime.legate_runtime.pop_provenance()

        with machine:
            task = self.legate_context.create_manual_task(
                LLMOpCode.HLO_EXECUTOR,
                launch_domain=launch_domain,
            )

            task.add_scalar_arg(self._next_run_id(), ty.uint64)
            task.add_scalar_arg(hlo_id, ty.uint64)
            task.add_scalar_arg(name, ty.string)
            for input in inputs:
                task.add_input(input)
            for red_op, output in zip(red_ops, outputs):
                if red_op is None:
                    task.add_output(output)
                    task.add_scalar_arg(0, ty.int8)
                else:
                    task.add_reduction(output, red_op)
                    task.add_scalar_arg(1, ty.int8)

            if machine.num_procs > 1:
                task.set_concurrent(True)

            task.execute()


runtime = LLMRuntime(llm_context)


class Tensor:
    def __init__(
        self,
        dtype,
        shape,
        hlo_id: int,
        name: str,
        optimize_scalar: bool = False,
    ):
        self.dtype = dtype
        self.shape = shape
        self.np_dtype = np.dtype(self.dtype.to_numpy_dtype())
        self.store = runtime.legate_context.create_store(
            self.dtype, self.shape, optimize_scalar=optimize_scalar
        )
        self.hlo_id = hlo_id
        self.name = name

        self.replicated = None
        self.partitions: dict[Shape, StorePartition] = {}

    @property
    def size(self) -> int:
        if self.replicated:
            return self.replicated.size

        itemsize = np.dtype(self.np_dtype).itemsize
        return np.prod(self.shape) * itemsize

    def replicate(self, ndevices) -> StorePartition:
        replicated_shape = (ndevices,) + self.shape
        tile_shape = (1,) + self.shape
        color_shape = (ndevices,) + (1,) * len(self.shape)
        if self.replicated is not None:
            return self.partitions[color_shape]

        self.replicated = Tensor(
            self.dtype,
            replicated_shape,
            self.hlo_id,
            self.name,
            optimize_scalar=False,
        )
        partition = self.replicated.store.partition_by_tiling(tile_shape)
        self.replicated.store.set_key_partition(partition.partition)
        self.partitions[color_shape] = partition
        return partition

    def partition_by_tiling(
        self, sharding: xla_data_pb2.OpSharding
    ) -> StorePartition:
        color_shape = tuple(sharding.tile_assignment_dimensions)
        partition = self.partitions.get(color_shape)
        if partition is not None:
            return partition

        if sharding.replicate_on_last_tile_dim:
            raise Exception(f"Bad sharding {sharding}")
            last_tile_dim = len(color_shape) - 1
            store = self.store.promote(
                last_tile_dim, color_shape[last_tile_dim]
            )
        else:
            store = self.store

        partition = store.partition_by_tiling(store.shape // color_shape)
        store.set_key_partition(partition.partition)

        self.partitions[color_shape] = partition
        return partition

    @staticmethod
    def create(instr: hlo_pb2.HloInstructionProto) -> Tensor:
        dtype = _CODES_TO_DTYPES[instr.shape.element_type]
        shape = tuple(instr.shape.dimensions)
        return Tensor(dtype, shape, instr.id, instr.name)


def shapes_equal(a, b):
    if a.element_type != b.element_type:
        return False

    if a.element_type == xla_data_pb2.PrimitiveType.TUPLE:
        for ai, bi in zip(a.tuple_shapes, b.tuple_shapes):
            if not shapes_equal(ai, bi):
                return False

    return True


class TensorSet:
    def __init__(self):
        self.microbatches: dict[int, Tensor] = {}
        self.last_microbatch = -1

    @property
    def size(self) -> int:
        size = 0
        for tensor in self.microbatches.values():
            size += tensor.size
        return size

    def get_input(
        self,
        input: hlo_pb2.HloInstructionProto,
        microbatch: Optional[int] = None,
    ) -> Tensor:
        if microbatch is None:
            # return whatever was the most recent microbatch
            microbatch = self.last_microbatch

        tensor = self.microbatches.get(microbatch)
        if tensor is None:
            raise Exception(
                f"cannot get microbatch {microbatch} for "
                f"input {input.name} that was never created"
            )
        return tensor

    def get_output(
        self,
        input: hlo_pb2.HloInstructionProto,
        microbatch: Optional[int] = None,
    ) -> Tensor:
        if microbatch is None:
            microbatch = 0
        self.last_microbatch = microbatch
        tensor = self.microbatches.get(microbatch)
        if tensor is None:
            # for outputs we can make a new tensor
            tensor = Tensor.create(input)
            self.microbatches[microbatch] = tensor
        # print(f"make {input.name} on mb={microbatch} -> {id(tensor)}")
        return tensor


class TensorMap:
    def __init__(self):
        self.external_tensors: dict[int, Tensor] = {}
        self.internal_tensors: dict[int, TensorSet] = {}
        self.aliases: dict[int, int] = {}

    @property
    def size(self) -> int:
        size = 0
        for tensor in self.external_tensors.values():
            size += tensor.size
        for tensor_set in self.internal_tensors.values():
            size += tensor_set.size
        return size

    def _get_external(self, instr: hlo_pb2.HloInstructionProto) -> Tensor:
        tensor = self.external_tensors.get(instr.id)
        if tensor is None:
            tensor = Tensor.create(instr)
            self.external_tensors[instr.id] = tensor
        return tensor

    def get_parameter(self, input: hlo_pb2.HloInstructionProto) -> Tensor:
        tensor = self._get_external(input)
        return tensor

    def find_alias(self, input_id: int) -> Optional[int]:
        return self.aliases.get(input_id)

    def add_parameter_alias(self, output_id: int, input_id: int) -> None:
        self.aliases[input_id] = output_id
        tensor = self.external_tensors.get(input_id)
        if tensor is None:
            raise Exception(
                f"cannot alias input {input_id} -> {output_id}, "
                f"{input_id} does not exist yet"
            )
        self.external_tensors[output_id] = tensor

    def get_input(
        self,
        input: hlo_pb2.HloInstructionProto,
        microbatch: Optional[int] = None,
    ) -> Tensor:
        tensor_set = self.internal_tensors.get(input.name)
        if tensor_set is None:
            tensor_set = TensorSet()
            self.internal_tensors[input.name] = tensor_set
        return tensor_set.get_input(input, microbatch)

    def get_output(
        self,
        output: hlo_pb2.HloInstructionProto,
        microbatch: Optional[int] = None,
    ) -> Tensor:
        tensor_set = self.internal_tensors.get(output.name)
        if tensor_set is None:
            tensor_set = TensorSet()
            self.internal_tensors[output.name] = tensor_set
        return tensor_set.get_output(output, microbatch)

    def get_root(self, root: hlo_pb2.HloInstructionProto) -> Tensor:
        tensor = self._get_external(root)
        return tensor


# Utility wrapper to generate a sub-module for a particular layer
class HloLayer:
    def __init__(
        self,
        key: LegateKey,
        computation_map: Optional[
            dict[int, hlo_pb2.HloComputationProto]
        ] = None,
    ):
        self.key = key
        self.entry_comp_id = None
        self.computation_map = computation_map
        self.contexts: dict[int, list[hlo_pb2.HloInstructionProto]] = {}
        self.comp_tree: Optional[dict[int, list[int]]] = None
        self.instructions_added = {}

        self.params = None
        self.inputs = None
        self.microbatch_inputs = None
        self.outputs = None
        self.microbatch_outputs = None
        self.roots = None
        self.max_instruction_id = 0

        self.remats = defaultdict(dict)

        self.pending_count = 0

        self.num_microbatches: Optional[int] = None
        self.microbatch_instr: Optional[hlo_pb2.HloInstructionProto] = None

    @property
    def interface_inputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.inputs + self.microbatch_inputs

    @property
    def interface_outputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.outputs + self.microbatch_outputs

    @property
    def batch_inputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.params + self.inputs

    @property
    def batch_outputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.roots + self.outputs

    @property
    def all_inputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.params + self.inputs + self.microbatch_inputs

    @property
    def all_outputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.roots + self.outputs + self.microbatch_outputs

    def set_microbatches(
        self,
        num_microbatches: int,
        microbatch_instr: hlo_pb2.HloInstructionProto,
    ) -> None:
        self.num_microbatches = num_microbatches
        self.microbatch_instr = microbatch_instr

    def add_parameter_aliases(self, aliased_params) -> None:
        for _, instructions in self.contexts.items():
            for instr in instructions:
                for operand_id in instr.operand_ids:
                    if (
                        operand_id not in self.instructions_added
                        and operand_id in aliased_params
                    ):
                        param_alias = aliased_params[operand_id]
                        if param_alias.tree is not None:
                            for tree_instr, tree_comp_id in param_alias.tree:
                                self.add_instruction(tree_comp_id, tree_instr)

    def _compute_tree(self, entry_comp_id: int, tree: list[int]):
        self.comp_tree[entry_comp_id] = tree[:]
        for instruction in self.contexts[entry_comp_id]:
            for comp_id in instruction.called_computation_ids:
                if comp_id in self.contexts:
                    next_tree = tree + [comp_id]
                    self.comp_tree[comp_id] = next_tree[:]
                    self._compute_tree(comp_id, next_tree)

    def compute_tree(self):
        if self.comp_tree is not None:
            return

        self.comp_tree = {}
        self._compute_tree(self.entry_comp_id, [self.entry_comp_id])
        for comp_id in self.contexts:
            if comp_id not in self.comp_tree:
                raise Exception(
                    f"failed to add comp {comp_id} to tree for {self.key}"
                )

    def _propagate_input(
        self, param: hlo_pb2.HloInstructionProto, tree: List[int]
    ):
        comp_id = tree[0]
        next_comp_id = tree[1] if len(tree) > 1 else None

        instructions = self.contexts[comp_id]
        instruction_map = {}
        tuple_arg_idx = None
        tuple_arg = None
        call_instr = None
        for idx, instr in enumerate(instructions):
            instruction_map[instr.id] = instr
            if instr.opcode == "parameter" and is_tuple_shape(instr):
                tuple_arg = instr
                tuple_arg_idx = idx
            if (
                next_comp_id is not None
                and next_comp_id in instr.called_computation_ids
            ):
                call_instr = instr

        new_param = hlo_pb2.HloInstructionProto()
        new_param.CopyFrom(param)
        new_param.ClearField("operand_ids")
        # the parameter needs to be renumbered since the original
        # number will only be used into bottom-level computation
        if next_comp_id is not None:
            self.max_instruction_id += 1
            # I am sure there is a better way to do this
            # but hack this for now
            new_param.id += comp_id * 10000  # self.max_instruction_id

        if tuple_arg is None:
            new_param.opcode = "parameter"
            # add as a simple parameter
            instructions.insert(0, new_param)
        else:
            new_param.opcode = "get-tuple-element"
            new_param.operand_ids.append(tuple_arg.id)
            new_param.tuple_index = len(tuple_arg.shape.tuple_shapes)
            tuple_arg.shape.tuple_shapes.append(new_param.shape)
            instructions.insert(tuple_arg_idx + 1, new_param)

        if call_instr is not None:
            first_arg = instruction_map[call_instr.operand_ids[0]]
            if (
                len(call_instr.operand_ids) == 1
                and first_arg.opcode == "tuple"
            ):
                # add the parameter to the input tuple
                first_arg.operand_ids.append(new_param.id)
                first_arg.shape.tuple_shapes.append(new_param.shape)
            else:
                # call or other with enumerated parameters not tupled
                call_instr.operand_ids.append(new_param.id)

        if next_comp_id is not None:
            self._propagate_input(param, tree[1:])

        return new_param

    def propagate_input(
        self, comp_id: int, param: hlo_pb2.HloInstructionProto
    ) -> hlo_pb2.HloInstructionProto:
        tree = self.comp_tree[comp_id]
        return self._propagate_input(param, tree)

    def compute_inputs(
        self,
        all_instructions: Mapping[int, int],
        param_map: Mapping[int, hlo_pb2.HloInstructionProto],
    ) -> Set[int]:
        self.compute_tree()

        inputs_needed = {}
        params_needed = set()
        for comp_id, instructions in self.contexts.items():
            comp_params = set()
            for instr in instructions:
                if instr.opcode == "parameter":
                    comp_params.add(instr.id)
                elif (
                    instr.opcode == "get-tuple-element"
                    and instr.operand_ids[0] in comp_params
                ):
                    # get tuple elements of the tuple arg should not have their
                    # operands counted as inputs, they are parameters
                    if instr.id in param_map:
                        params_needed.add(instr.id)
                    continue

                for operand_id in instr.operand_ids:
                    if operand_id in param_map:
                        params_needed.add(operand_id)
                    elif operand_id not in self.instructions_added:
                        inputs_needed[operand_id] = comp_id

        self.params = [param_map[id] for id in params_needed]

        self.inputs = []
        self.microbatch_inputs = []
        for operand_id, comp_id in inputs_needed.items():
            operand = all_instructions[operand_id]
            if comp_id == self.entry_comp_id:
                self.inputs.append(operand)
            else:
                self.microbatch_inputs.append(
                    self.propagate_input(comp_id, operand)
                )

        return inputs_needed

    def _propagate_output(
        self, output: hlo_pb2.HloInstructionProto, tree: List[int]
    ) -> hlo_pb2.HloComputationProto:
        comp_id = tree[0]
        next_comp_id = tree[1] if len(tree) > 1 else None

        instructions = self.contexts[comp_id]
        comp = self.computation_map[comp_id]
        new_output = output
        for instr in instructions:
            if next_comp_id in instr.called_computation_ids:
                # this needs to be added/extracted from the root tuple
                new_output = hlo_pb2.HloInstructionProto()
                new_output.CopyFrom(output)
                self.max_instruction_id += 1
                new_output.id += comp_id * 10000  # self.max_instruction_id
                new_output.opcode = "get-tuple-element"
                new_output.tuple_index = len(instr.shape.tuple_shapes)
                new_output.ClearField("operand_ids")
                new_output.operand_ids.append(instr.id)
                instr.shape.tuple_shapes.append(new_output.shape)
                instructions.insert(-1, new_output)
            elif instr.id == comp.root_id:
                instr.operand_ids.append(new_output.id)
                instr.shape.tuple_shapes.append(new_output.shape)

        if next_comp_id is not None:
            self._propagate_output(output, tree[1:])

        return new_output

    def propagate_output(
        self, comp_id: int, output: hlo_pb2.HloInstructionProto
    ) -> hlo_pb2.HloInstructionProto:
        tree = self.comp_tree[comp_id]
        # if this is already in the entry computation, there is nothing to do
        # we can immediately use this as an output
        if comp_id == self.entry_comp_id:
            return output

        return self._propagate_output(output, tree)

    def recompute_outputs(self, outputs_produced: Set[int]):
        def _prune(arr):
            new_arr = []
            for instr in arr:
                if instr.id not in outputs_produced:
                    new_arr.append(instr)
                    outputs_produced.add(instr.id)
            return new_arr

        self.roots = _prune(self.roots)
        self.outputs = _prune(self.outputs)
        self.microbatch_outputs = _prune(self.microbatch_outputs)

    def compute_outputs(
        self,
        all_outputs: Set[int],
        root_map: Mapping[int, hlo_pb2.HloInstructionProto],
    ) -> None:
        self.compute_tree()

        # do not add any inputs as an output
        input_names = set([input.name for input in self.all_inputs])

        self.outputs = []
        self.microbatch_outputs = []
        self.roots = []
        for comp_id, instructions in self.contexts.items():
            for instr in instructions:
                if instr.name not in input_names:
                    if instr.id in root_map:
                        self.roots.append(instr)
                    elif instr.id in all_outputs:
                        if comp_id == self.entry_comp_id:
                            self.outputs.append(instr)
                        else:
                            self.microbatch_outputs.append(
                                self.propagate_output(comp_id, instr)
                            )

    def __str__(self) -> str:
        str_arr = ["Layer %s" % self.key]
        if self.params:
            str_arr.append("  Parameters")
            for param in self.params:
                str_arr.append(
                    f"    {param.name:25} id={param.id:4} op={param.opcode:18} {param.shape.dimensions}"  # noqa: E501
                )
        if self.inputs:
            str_arr.append("  Inputs")
            for input in self.inputs:
                str_arr.append(
                    f"    {input.name:25} id={input.id:4} op={input.opcode:18} {input.shape.dimensions}"  # noqa: E501
                )
        if self.microbatch_inputs:
            str_arr.append("  Microbatch Inputs")
            for input in self.microbatch_inputs:
                str_arr.append(
                    f"    {input.name:25} id={input.id:4} op={input.opcode:18} {input.shape.dimensions}"  # noqa: E501
                )
        for id, context in self.contexts.items():
            str_arr.append(f"  Comp: {id}")
            for instr in context:
                if instr.opcode == "get-tuple-element":
                    opcode_str = f"get-element({instr.operand_ids[0]}[{instr.tuple_index}])"  # noqa: E501
                elif instr.opcode == "tuple":
                    opcode_str = f"tuple[{len(instr.shape.tuple_shapes)}]"
                elif instr.opcode == "parameter" and is_tuple_shape(instr):
                    opcode_str = (
                        f"tuple_parameter[{len(instr.shape.tuple_shapes)}]"
                    )
                elif is_tuple_shape(instr):
                    opcode_str = (
                        f"{instr.opcode}[{len(instr.shape.tuple_shapes)}]"
                    )
                else:
                    opcode_str = instr.opcode
                str_arr.append(
                    f"    {instr.name:25} id={instr.id:4} "
                    f"op={opcode_str:25} -> {instr.operand_ids}   {instr.shape.dimensions}"  # noqa: E501
                )
        if self.outputs:
            str_arr.append("  Outputs")
            for out in self.outputs:
                str_arr.append(
                    f"    {out.name:25} id={out.id:4} op={out.opcode:18} {out.shape.dimensions}"  # noqa: E501
                )
        if self.microbatch_outputs:
            str_arr.append("  Microbatch Outputs")
            for out in self.microbatch_outputs:
                str_arr.append(
                    f"    {out.name:25} id={out.id:4} op={out.opcode:18} {out.shape.dimensions}"  # noqa: E501
                )
        if self.roots:
            str_arr.append("  Roots")
            for root in self.roots:
                str_arr.append(
                    f"    {root.name:25} id={root.id:4} op={root.opcode}"
                )
        return "\n".join(str_arr)

    def compress_tuples(self):
        all_instructions = {}
        visited = set()

        class CompressedComputation:
            def __init__(self):
                self.params_kept = set()
                self.param_reindex = None
                self.root_reindex = None

        def _visit(comp_id):
            visited.add(comp_id)
            comp = self.computation_map[comp_id]
            tuple_param = None
            instructions = self.contexts[comp_id]
            compressed_comp = CompressedComputation()

            call_input_tuples: dict[int, CompressedComputation] = {}
            call_output_tuples: dict[int, CompressedComputation] = {}
            compressed_comps: dict[int, CompressedComputation] = {}

            def _add_comp_tuple(instr_id: int, tuple_id: int, comp_id: int):
                if comp_id in self.contexts:
                    compressed_comp = compressed_comps[comp_id]
                    if tuple_id is not None:
                        call_input_tuples[tuple_id] = compressed_comp
                    call_output_tuples[instr_id] = compressed_comp

            call_input_tuples = {}
            tuple_elements_found = {}
            for instr in instructions:
                all_instructions[instr.id] = instr
                for id in instr.called_computation_ids:
                    if id in self.contexts and id not in visited:
                        compressed_comps[id] = _visit(id)

                if instr.opcode == "tuple":
                    tuple_elements_found[instr.id] = []

                elif instr.opcode == "opt-barrier" and is_tuple_shape(instr):
                    tuple_elements_found[instr.id] = []
                    # the opt-barrier aliases a tuple,
                    # they should be compressed in the same way
                    tuple_elements_found[
                        instr.operand_ids[0]
                    ] = tuple_elements_found[instr.id]

                elif instr.opcode == "parameter":
                    if is_tuple_shape(instr):
                        tuple_elements_found[instr.id] = []
                        tuple_param = instr
                    compressed_comp.params_kept.add(instr.parameter_number)

                elif instr.opcode == "while":
                    _add_comp_tuple(
                        instr.id,
                        instr.operand_ids[0],
                        instr.called_computation_ids[0],
                    )
                    tuple_elements_found[instr.id] = []

                elif instr.opcode == "call":
                    if is_tuple_shape(instr):
                        tuple_elements_found[instr.id] = []

                    input_id = instr.operand_ids[0]
                    input = all_instructions.get(input_id)

                    called_comp_id = instr.called_computation_ids[0]
                    # if missing, this is not part of the layer which means
                    # that this cannot be an input tuple
                    if input is not None and is_tuple_shape(input):
                        _add_comp_tuple(instr.id, input.id, called_comp_id)
                    else:
                        _add_comp_tuple(instr.id, None, called_comp_id)
                        called_compressed_comp = compressed_comps.get(
                            called_comp_id
                        )
                        if called_compressed_comp is not None:
                            # if the input shape is not a tuple, see if we
                            # deleted any parameters and if so, fix them
                            if len(called_compressed_comp.params_kept) != len(
                                instr.operand_ids
                            ):
                                raise Exception(
                                    f"{instr.name} lost params in {self.key}"
                                )
                                new_ids = []
                                for idx, id in enumerate(instr.operand_ids):
                                    if (
                                        idx
                                        in called_compressed_comp.params_kept
                                    ):
                                        new_ids.append(id)
                                instr.ClearField("operand_ids")
                                instr.operand_ids.extend(new_ids)

                elif instr.opcode == "conditional":
                    true_comp_id, false_comp_id = instr.called_computation_ids
                    _, true_tuple_id, false_tuple_id = instr.operand_ids
                    _add_comp_tuple(instr.id, true_tuple_id, true_comp_id)
                    _add_comp_tuple(instr.id, false_tuple_id, false_comp_id)
                    tuple_elements_found[instr.id] = []

                    if instr.id not in call_output_tuples:
                        raise Exception(
                            f"conditional body not found in {self.key} "
                            "for comps {true_comp_id}, {false_comp_id}"
                        )

                elif instr.opcode == "get-tuple-element":
                    elements = tuple_elements_found.get(instr.operand_ids[0])
                    if elements is not None:
                        elements.append(instr)

            def _compress_tuple(reindex, tuple):
                # if length is the same, this is not compressed
                if reindex is None or len(reindex) == len(
                    tuple.shape.tuple_shapes
                ):
                    return

                new_operand_ids = [0] * len(reindex)
                new_shapes = [None] * len(reindex)
                is_tuple_instruction = tuple.opcode == "tuple"
                for old_index, new_index in reindex.items():
                    new_shapes[new_index] = tuple.shape.tuple_shapes[old_index]
                    if is_tuple_instruction:
                        # only tuple instructions should have their operands
                        # also compressed. Other instructions like while/call
                        # will be implicitly compressed by the call root.
                        new_operand_ids[new_index] = tuple.operand_ids[
                            old_index
                        ]

                if None in new_shapes:
                    raise Exception(
                        f"did not fill in all shapes for {tuple.name}"
                    )
                tuple.shape.ClearField("tuple_shapes")
                tuple.shape.tuple_shapes.extend(new_shapes)
                if is_tuple_instruction:
                    tuple.ClearField("operand_ids")
                    tuple.operand_ids.extend(new_operand_ids)

            def _reindex_elements(tuple_id, reindex=None):
                elements = tuple_elements_found.get(tuple_id)
                if elements is not None:
                    if reindex is None:
                        reindex = {}
                        elements.sort(key=lambda x: x.tuple_index)
                        for new_idx, element in enumerate(elements):
                            reindex[element.tuple_index] = new_idx
                            element.tuple_index = new_idx
                        return reindex
                    else:
                        for element in elements:
                            element.tuple_index = reindex[element.tuple_index]
                        return reindex
                return {}

            for tuple_id in tuple_elements_found.keys():
                tuple = all_instructions[tuple_id]
                tuple_comp_input = call_input_tuples.get(tuple.id)
                tuple_comp_output = call_output_tuples.get(tuple.id)
                if tuple_comp_input is not None:
                    reindex = tuple_comp_input.param_reindex
                    _reindex_elements(tuple_id, reindex)
                elif tuple_comp_output is not None:
                    reindex = tuple_comp_output.root_reindex
                    _reindex_elements(tuple_id, reindex)
                elif tuple_param is not None and tuple_param.id == tuple.id:
                    reindex = _reindex_elements(tuple_id)
                    compressed_comp.param_reindex = reindex
                else:
                    if tuple.opcode == "opt-barrier":
                        # there is no reindex, we simply copy the new shape
                        # of the operand to the instruction
                        # the input tuple has already been reindexed
                        input_tuple = all_instructions[tuple.operand_ids[0]]
                        tuple.shape.CopyFrom(input_tuple.shape)
                    else:
                        if tuple.id == comp.root_id:
                            reindex = {}
                            num_ids = 0
                            for idx, operand_id in enumerate(
                                tuple.operand_ids
                            ):
                                if operand_id in self.instructions_added:
                                    reindex[idx] = num_ids
                                    num_ids += 1
                            compressed_comp.root_reindex = reindex
                        else:
                            reindex = _reindex_elements(tuple_id, reindex=None)

                _compress_tuple(reindex, tuple)

            return compressed_comp

        if self.entry_comp_id is None:
            raise Exception(f"{self.key} has no entry computation")
        _visit(self.entry_comp_id)

    def noop(self) -> bool:
        if self.entry_comp_id is None:
            return True

        instructions = self.contexts[self.entry_comp_id]
        for instr in instructions:
            if instr.opcode not in ["parameter", "get-tuple-element"]:
                return False
        return True

    def split_microbatches(self) -> List[HloLayer]:
        layers = []
        new_key = self.key.replace(name=self.key.name + ".microbatch")
        microbatch_layer = HloLayer(new_key, self.computation_map)
        new_key = self.key.replace(name=self.key.name + ".post-microbatch")
        post_layer = HloLayer(new_key, self.computation_map)

        entry_comp = self.computation_map[self.entry_comp_id]

        all_instructions = {}

        instructions = self.contexts[self.entry_comp_id]

        def _add_context(layer, comp_id):
            instructions = self.contexts.get(comp_id)
            if instructions is not None:
                for instr in instructions:
                    layer.add_instruction(comp_id, instr)
                    for subcomp_id in instr.called_computation_ids:
                        _add_context(layer, subcomp_id)

        microbatch_root_tuple = None
        microbatch_while_id = None
        needed_instructions = set()
        num_microbatches = None
        num_mb_regex = re.compile(r"microbatches=(\d+)")
        for instr in instructions:
            all_instructions[instr.id] = instr
            if (
                instr.opcode == "while"
                and "microbatches=" in instr.metadata.op_name
            ):
                match = num_mb_regex.search(instr.metadata.op_name)
                if match is None:
                    raise Exception(
                        f"could not find number of microbatches for "
                        f"{instr.name} in {instr.metadata.op_name}"
                    )
                num_microbatches = int(match.groups()[0])
                microbatch_counter_instr = all_instructions[
                    instr.operand_ids[0]
                ]

                microbatch_layer.set_microbatches(
                    num_microbatches, microbatch_counter_instr
                )

                # microbatch_layer.add_instruction(self.entry_comp_id, instr)
                # we have to convert the while loop into a call instruction
                while_call_instr = hlo_pb2.HloInstructionProto()
                while_call_instr.opcode = "call"
                while_call_instr.id = instr.id
                while_call_instr.name = instr.name
                body_id = instr.called_computation_ids[0]
                while_call_instr.called_computation_ids.append(body_id)
                while_call_instr.operand_ids.append(instr.operand_ids[0])
                while_call_instr.shape.CopyFrom(instr.shape)
                microbatch_layer.add_instruction(
                    self.entry_comp_id, while_call_instr
                )

                _add_context(microbatch_layer, body_id)
                layers.append(microbatch_layer)

                microbatch_root_tuple = hlo_pb2.HloInstructionProto()
                microbatch_root_tuple.CopyFrom(instr)
                microbatch_root_tuple.shape.ClearField("tuple_shapes")
                microbatch_root_tuple.ClearField("operand_ids")
                microbatch_root_tuple.ClearField("called_computation_ids")
                microbatch_root_tuple.opcode = "tuple"
                microbatch_root_tuple.id = entry_comp.root_id
                microbatch_while_id = instr.id
            else:
                if num_microbatches is None:
                    match = num_mb_regex.search(instr.metadata.op_name)
                    if match is not None:
                        num_microbatches = int(match.groups()[0])

                # all get tuple elements of the microbatching while
                # loop should be rolled into the microbatch part to
                # make a new root tuple
                if (
                    instr.opcode == "get-tuple-element"
                    and instr.operand_ids[0] == microbatch_while_id
                ):
                    microbatch_layer.add_instruction(self.entry_comp_id, instr)
                    microbatch_root_tuple.shape.tuple_shapes.append(
                        instr.shape
                    )
                    microbatch_root_tuple.operand_ids.append(instr.id)
                else:
                    needed_instructions.add(instr.id)
                    if microbatch_root_tuple is None:
                        current_layer = microbatch_layer
                    else:
                        # make sure operands before the split are added
                        # to the post layer
                        for operand_id in instr.operand_ids:
                            if operand_id in needed_instructions:
                                operand = all_instructions[operand_id]
                                post_layer.add_instruction(
                                    self.entry_comp_id, operand
                                )
                        current_layer = post_layer

                    current_layer.add_instruction(self.entry_comp_id, instr)
                    for comp_id in instr.called_computation_ids:
                        _add_context(current_layer, comp_id)

        if microbatch_root_tuple is None:
            # this might still be part of the microbatching even if
            # it doesn't receive any inputs from the while loop or
            # produce any outputs to the while loop
            if num_microbatches is not None:
                # this has to be independent of the microbatch loop index
                # otherwise the while loop would have been a dependent
                self.set_microbatches(num_microbatches, None)
            return (self,)

        if post_layer.noop():
            return (microbatch_layer,)

        microbatch_layer.add_instruction(
            self.entry_comp_id, microbatch_root_tuple
        )
        return microbatch_layer, post_layer

    def add_rematerialization(
        self, comp_id: int, instr: hlo_pb2.HloInstructionProto
    ) -> None:
        instr_map = self.remats[comp_id]
        instr_map[instr.id] = instr

    def add_instruction(
        self,
        comp_id: int,
        instr: hlo_pb2.HloInstructionProto,
        all_instructions: Optional[
            Mapping[int.hlo_pb2.HloInstructionProto]
        ] = None,
        depth=0,
    ):
        if instr.id in self.instructions_added:
            return

        if self.entry_comp_id is None:
            self.entry_comp_id = comp_id

        def _is_derived_constant(_instr, recursion: int = 0) -> bool:
            if recursion > 5:
                # quit searching after going back 5 operands
                return False

            if len(_instr.operand_ids) == 0:
                return _instr.opcode == "constant"

            for operand_id in _instr.operand_ids:
                operand = all_instructions[operand_id]
                if not _is_derived_constant(operand, recursion + 1):
                    return False

            return True

        if (
            not is_tuple_shape(instr) and all_instructions is not None
        ):  # and instr.opcode in _
            for operand_id in instr.operand_ids:
                operand = all_instructions[operand_id]

                if instr.opcode == "broadcast":
                    # see if the operand is smaller, if so add it
                    # to shrink the surface area
                    instr_size = np.prod(instr.shape.dimensions)
                    operand_size = np.prod(operand.shape.dimensions)
                    if operand_size < instr_size:
                        self.add_instruction(
                            comp_id, operand, all_instructions
                        )
                        continue

                if _is_derived_constant(operand):
                    self.add_instruction(comp_id, operand, all_instructions)
                    continue

                if operand.opcode in ["convert", "bitcast", "constant"]:
                    self.add_instruction(comp_id, operand, all_instructions)
                    continue

        # add 1000 to avoid any weird conflicts
        self.max_instruction_id = max(self.max_instruction_id, instr.id + 1000)

        if instr.id not in self.instructions_added:
            comp = self.computation_map[comp_id]
            if instr.id != comp.root_id:
                # do not rematerialize backwards from the root
                # we do not want to add new outputs that we don't need
                remats = self.remats.get(comp_id)
                if remats is not None:
                    for operand_id in instr.operand_ids:
                        # recurse the rematerialization tree to add all
                        # previous instructions that should be recomputed
                        remat = remats.get(operand_id)
                        if remat is not None:
                            self.add_instruction(
                                comp_id, remat, depth=depth + 1
                            )

            instr_list = self.contexts.get(comp_id)
            if instr_list is None:
                if self.comp_tree is not None:
                    raise Exception(
                        f"cannot add new computation {comp_id} for "
                        f"{instr.name}, tree already computed"
                    )
                self.contexts[comp_id] = []
                instr_list = self.contexts[comp_id]

            copy_instr = hlo_pb2.HloInstructionProto()
            copy_instr.CopyFrom(instr)
            instr_list.append(copy_instr)
            self.instructions_added[instr.id] = copy_instr

    def _generate_hlo_module_proto(
        self, tensors: TensorMap
    ) -> hlo_pb2.HloModuleProto:
        hlo_module = hlo_pb2.HloModuleProto()
        hlo_module.name = str(self.key)

        entry_comp = hlo_pb2.HloComputationProto()
        entry_comp.id = self.entry_comp_id
        entry_comp.name = "%s_main.%d" % (self.key, self.entry_comp_id)

        self.max_instruction_id += 1
        root_id = self.max_instruction_id

        program_shape = xla_data_pb2.ProgramShapeProto()

        parameter_number = 0
        parameter_names: OrderedSet[str] = OrderedSet()

        def add_parameter(
            param: hlo_pb2.HloInstructionProto, parameter_number: int
        ) -> int:
            parameter_names.add(param.name)
            param.parameter_number = parameter_number
            entry_comp.instructions.append(param)
            program_shape.parameters.append(param.shape)
            program_shape.parameter_names.append(param.name)
            return parameter_number + 1

        for input in self.all_inputs:
            # if the input comes from another layer
            # in the parent HLO module, this will be an add,dot,etc
            # we keep the original name, but change it to a parameter
            new_param = hlo_pb2.HloInstructionProto()
            new_param.CopyFrom(input)
            new_param.ClearField("operand_ids")
            new_param.ClearField("called_computation_ids")
            new_param.opcode = "parameter"
            # clear the op name, parameters do not have metadata op names
            # in HLO modules. make this consistent with regular Jax.
            parameter_number = add_parameter(new_param, parameter_number)

        inout_alias = hlo_pb2.HloInputOutputAliasProto()
        all_outputs = self.roots + self.outputs + self.microbatch_outputs
        for param_idx, param in enumerate(self.params):
            matching_root_id = tensors.find_alias(param.id)
            if matching_root_id is not None:
                matching_root_idx = None
                for root_idx, root in enumerate(self.roots):
                    if root.id == matching_root_id:
                        matching_root_idx = root_idx
                        break
                if matching_root_idx is not None:
                    entry = hlo_pb2.HloInputOutputAliasProto.AliasEntryProto()
                    if len(all_outputs) > 1:
                        entry.output_shape_index.append(root_idx)
                    # entry.parameter_shape_index.append(param_idx)
                    entry.parameter_number = param_idx
                    entry.kind = hlo_pb2.Kind.MUST_ALIAS
                    inout_alias.entries.append(entry)

        if len(inout_alias.entries) > 0:
            hlo_module.input_output_alias.CopyFrom(inout_alias)

        _make_subcomp = None
        _visit_subcomp = None
        comps_added = set()
        instructions_added = set()

        def _add_instruction(
            comp: hlo_pb2.HloComputationProto,
            instr: hlo_pb2.HloInstructionProto,
        ):
            if instr.id in instructions_added:
                return False

            if comp.id == self.entry_comp_id and (
                instr.name in parameter_names or instr.opcode == "parameter"
            ):
                return False

            for operand_id in instr.operand_ids:
                operand = self.instructions_added.get(operand_id)
                # this is part of this layer, make sure it is added
                # before the user instruction
                if operand is not None:
                    _add_instruction(comp, operand)

            comp.instructions.append(instr)
            instructions_added.add(instr.id)
            for id in instr.called_computation_ids:
                if id not in self.contexts:
                    _visit_subcomp(id)
                elif id not in comps_added:
                    subcomp = _make_subcomp(id)
                    hlo_module.computations.append(subcomp)
                    comps_added.add(id)

            return True

        def _visit_subcomp(comp_id: int):
            if comp_id not in comps_added:
                subcomp = self.computation_map[comp_id]
                for instr in subcomp.instructions:
                    for id in instr.called_computation_ids:
                        _visit_subcomp(id)
                hlo_module.computations.append(subcomp)
                comps_added.add(comp_id)

        def _make_subcomp(comp_id: int):
            comp = hlo_pb2.HloComputationProto()
            comp.id = comp_id
            comp.name = f"{self.key}.comp.{comp_id}"
            instructions = self.contexts[comp_id]
            for instr in instructions:
                _add_instruction(comp, instr)
            comp.root_id = instructions[-1].id
            return comp

        entry_comp_instructions = self.contexts[self.entry_comp_id]
        for instr in entry_comp_instructions:
            _add_instruction(entry_comp, instr)

        if len(all_outputs) == 1:
            # If there is a single output from this module, make that single
            # HLO instruction the ROOT (i.e. return value)
            entry_comp.root_id = all_outputs[0].id
            # The program result shape is the shape of the single HLO
            program_shape.result.CopyFrom(all_outputs[0].shape)
        else:
            # There are multiple outputs from this module, we need to combine
            # them into a tuple ROOT (i.e. return value)
            root_instruction = hlo_pb2.HloInstructionProto()
            root_instruction.id = root_id
            root_instruction.name = "new_root.%d" % root_id
            root_instruction.opcode = "tuple"
            root_shape = xla_data_pb2.ShapeProto()
            root_shape.element_type = xla_data_pb2.PrimitiveType.TUPLE
            for output in all_outputs:
                root_shape.tuple_shapes.append(output.shape)
                root_instruction.operand_ids.append(output.id)
            root_instruction.shape.CopyFrom(root_shape)
            # The program result shape is the tuple shape of all return values
            program_shape.result.CopyFrom(root_shape)
            entry_comp.instructions.append(root_instruction)
            entry_comp.root_id = root_id

        entry_comp.program_shape.CopyFrom(program_shape)
        hlo_module.entry_computation_id = entry_comp.id
        hlo_module.computations.append(entry_comp)
        hlo_module.host_program_shape.CopyFrom(program_shape)

        # validate the module in a few different ways
        # 1) Make sure no computation or instruction has been added twice
        # 2) Make sure that ordering of instructions/comptuations is correct.
        #    All called computations of an instruction must precede that
        #    instruction in the list.  All operands of an inustrction must
        #    precede that instruction in the list.
        all_computation_ids = {}
        all_instruction_ids = {}
        all_instruction_names = {}
        for comp in hlo_module.computations:
            if comp.id in all_computation_ids:
                raise RuntimeError(
                    f"Computation {comp.name} got added twice in {self.key}"
                )
            all_computation_ids[comp.id] = comp

            params_seen = {}

            for instr in comp.instructions:
                if instr.opcode == "parameter":
                    prev = params_seen.get(instr.parameter_number)
                    if prev is not None:
                        raise Exception(
                            f"{comp.id} in {self.key} has multiple parameter "
                            f"no. {instr.parameter_number}: "
                            f"{prev.name} and {instr.name} in {comp.name}"
                        )
                    params_seen[instr.parameter_number] = instr

                if instr.opcode == "get-tuple-element":
                    operand = all_instruction_ids[instr.operand_ids[0]]
                    tuple_shape = operand.shape.tuple_shapes[instr.tuple_index]
                    if tuple_shape.element_type != instr.shape.element_type:
                        raise Exception(
                            f"{instr.name} shape mismatch {operand.name}"
                            f" on index {instr.tuple_index}"
                            f" in {comp.name}"
                        )

                if instr.opcode == "tuple":
                    for idx, operand_id in enumerate(instr.operand_ids):
                        operand = all_instruction_ids[operand_id]
                        tuple_shape = instr.shape.tuple_shapes[idx]
                        if (
                            tuple_shape.element_type
                            != operand.shape.element_type
                        ):
                            raise Exception(
                                f"{instr.name} shape mismatch {operand.name}"
                                f" on index {idx}"
                            )

                if instr.opcode == "call":
                    subcomp = all_computation_ids[
                        instr.called_computation_ids[0]
                    ]
                    params = [
                        instr
                        for instr in subcomp.instructions
                        if instr.opcode == "parameter"
                    ]
                    params.sort(key=lambda x: x.parameter_number)
                    operands = [
                        all_instruction_ids[id] for id in instr.operand_ids
                    ]
                    for param, operand in zip(params, operands):
                        if not shapes_equal(param.shape, operand.shape):
                            raise Exception(
                                f"{param.name} shape mismatch {operand.name}"
                                f" in {subcomp.name}:"
                                f"\n{param.shape}\n{operand.shape}"
                            )

                    root = all_instruction_ids[subcomp.root_id]
                    if not shapes_equal(root.shape, instr.shape):
                        raise Exception(
                            f"{instr.name} differs from root {root.name}"
                            f" in {comp.name}\n{param.shape}\n{operand.shape}"
                        )

                if instr.opcode == "conditional":

                    def _get_arg_tuple(_comp):
                        for _instr in _comp.instructions:
                            if _instr.opcode == "parameter":
                                return _instr

                    def _check_comp(_instr, operand_id, comp_id):
                        operand = all_instruction_ids[operand_id]
                        _comp = all_computation_ids[comp_id]
                        arg_tuple = _get_arg_tuple(_comp)
                        if not shapes_equal(arg_tuple.shape, operand.shape):
                            raise Exception(
                                f"{param.name} shape mismatch {operand.name}"
                                f"in {_comp.name}:\n{param.shape}"
                                f"\n{operand.shape}"
                            )
                        root = all_instruction_ids[_comp.root_id]
                        if not shapes_equal(root.shape, _instr.shape):
                            raise Exception(
                                f"{root.name} shape mismatch {_instr.name}"
                                f"in {_comp.name}:\n{root.shape}"
                                f"\n{_instr.shape}"
                            )

                    _check_comp(
                        instr,
                        instr.operand_ids[1],
                        instr.called_computation_ids[0],
                    )
                    _check_comp(
                        instr,
                        instr.operand_ids[2],
                        instr.called_computation_ids[1],
                    )

                if instr.name in all_instruction_names:
                    duplicate_instr = all_instruction_names[instr.name]
                    duplicate_instr.name += f".{duplicate_instr.id}"
                    if duplicate_instr.name in all_instruction_names:
                        raise Exception(
                            f"failed to uniquify name for {instr.name}"
                        )

                all_instruction_names[instr.name] = instr
                for operand_id in instr.operand_ids:
                    if operand_id not in all_instruction_ids:
                        raise RuntimeError(
                            f"Instruction {instr.name} is missing operand "
                            f"{operand_id} in {self.key}"
                        )

                for called_id in instr.called_computation_ids:
                    if called_id not in all_computation_ids:
                        raise RuntimeError(
                            f"Instruction {instr.name}:{instr.opcode} is "
                            f"missing called computation {called_id} in "
                            f"{self.key}:{comp.name}"
                        )
                if instr.id in all_instruction_ids:
                    raise RuntimeError(
                        f"Instruction {instr.name} id={instr.id} "
                        f"got added twice in {self.key}"
                    )

                if instr.opcode == "parameter" and len(instr.operand_ids) > 0:
                    raise RuntimeError(
                        f"{instr.name} parameter has operands in {self.key}"
                    )

                all_instruction_ids[instr.id] = instr

        return hlo_module

    def to_module(self, tensors: TensorMap) -> HloModule:
        hlo_module = self._generate_hlo_module_proto(tensors)
        return HloModule(
            self.key,
            parameters=self.params,
            inputs=self.inputs,
            microbatch_inputs=self.microbatch_inputs,
            roots=self.roots,
            outputs=self.outputs,
            microbatch_outputs=self.microbatch_outputs,
            hlo_module=hlo_module,
            num_microbatches=self.num_microbatches,
        )


class HloLayerQueue:
    def __init__(self, layers: list[HloLayer]):
        self.depends_on: dict[str, list[HloLayer]] = {}
        self.ready_layers = []

        for layer in layers:
            inputs = layer.interface_inputs
            if len(inputs) == 0:
                self.ready_layers.append(layer)
                continue
            for input in inputs:
                if input.name not in self.depends_on:
                    self.depends_on[input.name] = []
                self.depends_on[input.name].append(layer)
                layer.pending_count += 1

    def pop_ready(self) -> Optional[HloLayer]:
        if not self.ready_layers:
            return None
        return self.ready_layers.pop(0)

    def finish(self, layer: HloLayer):
        for output in layer.interface_outputs:
            if output.name not in self.depends_on:
                raise Exception(
                    f"{output.name} in {layer.key} "
                    "is not a dependency to any layer"
                )
            for dependent_layer in self.depends_on[output.name]:
                assert dependent_layer.pending_count > 0
                dependent_layer.pending_count -= 1
                if dependent_layer.pending_count == 0:
                    self.ready_layers.append(dependent_layer)
            del self.depends_on[output.name]


def find_derived_constant(
    root: hlo_pb2.HloInstructionProto,
    entry_def_map: dict[int, hlo_pb2.HloInstructionProto],
) -> Optional[hlo_pb2.HloInstructionProto]:
    if root.opcode == "constant":
        return root

    next = root
    while next.opcode in _CHEAP_REPLICATED_OPS:
        operand_id = next.operand_ids[0]
        next = entry_def_map[operand_id]
        if next.opcode == "constant":
            return next
    return None


def sharding_to_color_shape(
    sharding: xla_data_pb2.OpSharding,
) -> Optional[Shape]:
    if sharding.replicate_on_last_tile_dim:
        raise Exception("Replicated tensors not yet supported")

    if len(sharding.tile_assignment_dimensions) > 0:
        return Shape(sharding.tile_assignment_dimensions)
    else:
        return None


class ShardedHloModule:
    def __init__(
        self,
        hlo_module: hlo_pb2.HloModuleProto,
        input_shardings: Mapping[int, xla_data_pb2.OpSharding],
        output_shardings: Mapping[int, xla_data_pb2.OpSharding],
    ):
        self._hlo_module = hlo_module
        self._input_shardings = input_shardings
        self._output_shardings = output_shardings
        self._id = -1

    @property
    def hlo_module(self) -> hlo_pb2.HloModuleProto:
        return self._hlo_module

    @property
    def input_shardings(self) -> Mapping[int, Shape]:
        return self._input_shardings

    @property
    def output_shardings(self) -> Mapping[int, Shape]:
        return self._output_shardings

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        self._id = value


def compute_module_sharding(
    hlo_module: hlo_pb2.HloModuleProto,
    inputs: Sequence[hlo_pb2.HloInstructionProto],
    outputs: Sequence[hlo_pb2.HloInstructionProto],
    mesh: TaskMesh,
) -> ShardedHloModule:
    input_ids = OrderedSet(input.id for input in inputs)
    output_ids = OrderedSet(output.id for output in outputs)

    input_shardings: Mapping[int, xla_data_pb2.OpSharding] = {}
    output_shardings: Mapping[int, xla_data_pb2.OpSharding] = {}

    def _shard(
        instr: hlo_pb2.HloInstructionProto,
        trace: Optional[Mapping[int, xla_data_pb2.OpSharding]] = None,
    ):
        mesh.shard(instr)
        if (
            trace is not None
            and len(instr.sharding.tile_assignment_dimensions) > 0
        ):
            trace[instr.id] = instr.sharding
        if instr.sharding.replicate_on_last_tile_dim:
            raise Exception(
                f"{instr.name} has unsupported replicated sharding in "
                f"module {hlo_module.name}: {instr.metadata.op_name}"
            )

    sharded_module = hlo_pb2.HloModuleProto()
    sharded_module.CopyFrom(hlo_module)
    entry_comp = find_entry_computation(sharded_module)

    tupled_arg_param: Optional[hlo_pb2.HloInstructionProto] = None
    input_parameters: List[hlo_pb2.HloInstructionProto] = []
    # for comp in sharded_module.computations:
    for instr in entry_comp.instructions:
        # only shard inputs and outputs
        # or intermediates that were explicitly marked
        # do not shard intermediates that have their logical
        # axes derived implicitly from sharding propagation
        if instr.opcode == "parameter" and is_tuple_shape(instr):
            # this must be the tupled args parameter
            tupled_arg_param = instr
        elif instr.id in input_ids:
            _shard(instr, input_shardings)
            input_parameters.append(instr)
        elif instr.id in output_ids:
            _shard(instr, output_shardings)
        elif instr.id == entry_comp.root_id and is_tuple_shape(instr):
            # XLA requires tuple sharding annotation on the root tuple
            tuple_sharding = xla_data_pb2.OpSharding()
            tuple_sharding.type = xla_data_pb2.OpSharding.Type.TUPLE
            for operand_id in instr.operand_ids:
                operand_sharding = output_shardings.get(operand_id)
                if operand_sharding is None:
                    operand_sharding = xla_data_pb2.OpSharding()
                    operand_sharding.type = (
                        xla_data_pb2.OpSharding.Type.REPLICATED
                    )
                tuple_sharding.tuple_shardings.append(operand_sharding)
            instr.sharding.CopyFrom(tuple_sharding)
        elif "implicit_axes" not in instr.metadata.op_name:
            _shard(instr)

    for comp in sharded_module.computations:
        # we have already processed the entry computation
        # skip it here
        if comp.id != entry_comp.id:
            for instr in comp.instructions:
                if "implicit_axes" not in instr.metadata.op_name:
                    _shard(instr)

    if tupled_arg_param is not None:
        tuple_sharding = xla_data_pb2.OpSharding()
        tuple_sharding.type = xla_data_pb2.OpSharding.Type.TUPLE
        input_parameters.sort(key=lambda instr: instr.tuple_index)
        for parameter in input_parameters:
            tuple_sharding.tuple_shardings.append(parameter.sharding)
        tupled_arg_param.sharding.CopyFrom(tuple_sharding)

    return ShardedHloModule(sharded_module, input_shardings, output_shardings)


def get_roots(
    entry_comp: hlo_pb2.HloInstructionProto,
    entry_def_map: dict[int, hlo_pb2.HloInstructionProto],
    prune_constants: bool = True,
) -> Tuple[list[hlo_pb2.HloInstructionProto], bool, Mapping[int, int]]:
    root_instr = entry_def_map[entry_comp.root_id]
    roots: list[hlo_pb2.HloInstructionProto] = []

    # keeps track of whether any root elements in the tuple are not
    # necessary to include
    found_constants = False
    # if any roots get pruned, the output tuple will need to be reindexed
    output_reindex = {}
    if root_instr.opcode == "tuple":
        for idx, operand_id in enumerate(root_instr.operand_ids):
            instr = entry_def_map[operand_id]
            # there are constant returns that serve no useful purpose
            # sometimes the return is a reshape/broadcast of that constant
            if prune_constants:
                parent_constant = find_derived_constant(instr, entry_def_map)
            else:
                parent_constant = None

            if parent_constant is None:
                # in case we prune any roots, we need to re-index
                # the root tuple to fill in the gaps
                output_reindex[idx] = len(roots)
                roots.append(instr)
            else:
                # this constant is never used by any real operations
                # we can prune it and save ourselves on runtime overhead
                found_constants = True
    else:
        roots.append(root_instr)

    return roots, found_constants, output_reindex


def get_parameters(entry_comp: hlo_pb2.HloInstructionProto):
    parameters = [
        instr
        for instr in entry_comp.instructions
        if instr.opcode == "parameter"
    ]
    parameters.sort(key=lambda instr: instr.parameter_number)
    if len(parameters) == 1 and is_tuple_shape(parameters[0]):
        param_id = parameters[0].id
        parameters = []
        for instr in entry_comp.instructions:
            if (
                instr.opcode == "get-tuple-element"
                and instr.operand_ids[0] == param_id
            ):
                parameters.append(instr)
            parameters.sort(key=lambda instr: instr.tuple_index)
    return parameters


def get_entry_tuple_arg(
    comp: hlo_pb2.HloComputationProto,
) -> hlo_pb2.HloInstructionProto:
    for instr in comp.instructions:
        if instr.opcode == "parameter":
            assert instr.shape.element_type == xla_data_pb2.PrimitiveType.TUPLE
            return instr


class InstructionColoring:
    def __init__(
        self,
        color: Optional[LegateKey] = None,
        tuple_colors: Optional[List[InstructionColoring]] = None,
    ):
        self.tuple_colors = (
            None if tuple_colors is None else list(tuple_colors)
        )
        self.colors = set()
        self.explicit = color
        if self.explicit and self.tuple_colors:
            raise Exception("have tuple colors and explict color")
        if self.explicit is not None:
            assert isinstance(color, LegateKey)

    @property
    def single_color(self) -> Optional[LegateKey]:
        if self.explicit:
            return self.explicit
        elif len(self.colors) == 1:
            return next(iter(self.colors))
        return None

    def _explicit_merge(self, other):
        single_color = other.single_color
        if single_color is not None:
            if single_color == self.explicit:
                return self, True
            return None, False  # these do not agree

        # the only other way we can say these agree
        # is for the other to be empty
        if other.empty:
            return self, False

    def merge(
        self, other: InstructionColoring
    ) -> Optional[InstructionColoring]:
        if self.explicit:
            return self._explicit_merge(other)
        if other.explicit:
            return other._explicit_merge(other)

        # for now, disallow merging if these are not
        # explicitly given a single color
        return None, False

    def __eq__(self, other):
        if self.single_color:
            if other.single_color:
                return self.single_color == other.single_color
            return False
        return False

    def __str__(self):
        if self.explicit:
            return str(self.explicit)
        elif self.tuple_colors is not None:
            return str(list(map(str, self.tuple_colors)))
        return str(list(self.colors))

    def empty(self) -> bool:
        if self.explicit:
            return False
        if self.tuple_colors is not None:
            return len(self.tuple_colors) == 0
        return len(self.colors) == 0

    def all_colors(self) -> list[LegateKey]:
        if self.explicit:
            return [self.explicit]
        if self.tuple_colors is not None:
            colors = []
            for color in self.tuple_colors:
                colors.extend(color.all_colors())
            return colors
        return list(self.colors)

    def copy(self) -> InstructionColoring:
        if self.explicit:
            return InstructionColoring(color=self.explicit)
        elif self.tuple_colors is not None:
            return InstructionColoring(tuple_colors=self.tuple_colors)
        color = InstructionColoring()
        color.colors = self.colors.copy()
        return color

    def append(self, color: InstructionColoring) -> None:
        if self.tuple_colors:
            for tuple_color in self.tuple_colors:
                tuple_color.append(color)
        else:
            if (
                self.explicit
                and color.explicit
                and self.explicit == color.explicit
            ):
                return  # nothing to do
            new_colors = self.all_colors() + color.all_colors()
            self.colors = set(new_colors)
            self.explicit = False

    def add_color(self, color: InstructionColoring) -> bool:
        if self.explicit:
            return False
        else:
            size = len(self.colors)
            for color in color.all_colors():
                self.colors.add(color)
            return size != len(self.colors)


def color_backwards(
    instr: hlo_pb2.HloInstructionProto,
    color: InstructionColoring,
    all_instructions: Mapping[int, hlo_pb2.HloInstructionProto],
    instruction_colors: dict[int, InstructionColoring],
):
    if instr.opcode == "get-tuple-element":
        tuple_operand = all_instructions[instr.operand_ids[0]]
        if tuple_operand.opcode in [
            "tuple",
            "opt-barrier",
            "parameter",
        ]:  # only propagate through tuples, not whiles, conditionals
            tuple_colors = instruction_colors[tuple_operand.id].tuple_colors
            tuple_index_color = tuple_colors[instr.tuple_index]
            if tuple_index_color.add_color(color):
                if tuple_operand.opcode != "parameter":
                    operand = tuple_operand.operand_ids[instr.tuple_index]
                    color_backwards(
                        operand, color, all_instructions, instruction_colors
                    )
    elif instr.opcode == "tuple":
        tuple_color = instruction_colors[instr.id]
        if tuple_color.tuple_colors is None:
            raise Exception(f"{instr.name} has no tuple colors")
        if color.tuple_colors is None:
            raise Exception(
                f"{color} is not a tuple, cannot propagate to {instr.name}"
            )
        for idx, (existing, assigned) in enumerate(
            zip(color.tuple_colors, tuple_color.tuple_colors)
        ):
            merged_color, equal = existing.merge(assigned)
            operand = all_instructions[instr.operand_ids[idx]]
            if not equal:
                tuple_color.tuple_colors[idx] = merged_color.copy()
                instruction_colors[operand.id] = merged_color.copy()
                color_backwards(
                    operand, merged_color, all_instructions, instruction_colors
                )
    elif instr.opcode == "opt-barrier":
        # this should just copy the color directly
        operand = all_instructions[instr.operand_ids[0]]
        color_backwards(operand, color, all_instructions, instruction_colors)
    else:
        for operand_id in instr.operand_ids:
            operand_color = instruction_colors[operand_id]
            if operand_color.add_color(color):
                operand = all_instructions[operand_id]
                color_backwards(
                    operand, color, all_instructions, instruction_colors
                )


class ParameterAlias:
    def __init__(self, alias=None, aliases=None, tree=None):
        self.alias = alias
        self.aliases = aliases
        if alias:
            assert tree is not None
            self.tree = tree
        else:
            self.tree = None

    def new_tree(self, instr, comp_id) -> ParameterAlias:
        if self.alias:
            new_tree = self.tree + [(instr, comp_id)]
            return ParameterAlias(alias=self.alias, tree=new_tree)

        new_aliases = [
            alias.new_tree(instr, comp_id) if alias is not None else None
            for alias in self.aliases
        ]
        return ParameterAlias(aliases=new_aliases)

    def tree_string(self) -> str:
        if self.tree is None:
            return " [] "
        str_arr = []
        for instr, comp_id in self.tree:
            str_arr.append(f"{instr.name} -> {comp_id}")
        return "\n".join(str_arr)

    def __str__(self) -> str:
        if self.alias is not None:
            if isinstance(self.alias, ParameterAlias):
                return str(self.alias)
            else:
                return self.alias.name
        else:
            alias_arr = [str(alias) for alias in self.aliases]
            return str(alias_arr)

    def set_index(self, idx, alias):
        self.aliases[idx] = alias

    def get_index(self, idx):
        return self.aliases[idx]


def compute_parameter_aliases(
    entry_comp: hlo_pb2.HloComputationProto,
    computation_map: dict[int, hlo_pb2.HloComputationProto],
    all_instructions: Mapping[int, hlo_pb2.HloInstructionProto],
    parameter_aliases=None,
):
    if parameter_aliases is None:
        parameter_aliases = {}
        params = get_parameters(entry_comp)
        for param in params:
            parameter_aliases[param.id] = ParameterAlias(
                alias=param, tree=[(param, entry_comp.id)]
            )

    def _add_subcomp_aliases(instr, comp_id, operand_ids):
        subcomp = computation_map[comp_id]
        params = [
            instr
            for instr in subcomp.instructions
            if instr.opcode == "parameter"
        ]
        params.sort(key=lambda x: x.parameter_number)
        for operand_id, param in zip(operand_ids, params):
            alias = parameter_aliases.get(operand_id)
            if alias is not None:
                parameter_aliases[param.id] = alias.new_tree(
                    instr, entry_comp.id
                )

        compute_parameter_aliases(
            subcomp, computation_map, all_instructions, parameter_aliases
        )

        # root_alias = parameter_aliases.get(subcomp.root_id)
        # if root_alias is not None:
        #    parameter_aliases[instr.id] = root_alias

    for instr in entry_comp.instructions:
        if instr.opcode in ["bitcast", "reshape", "convert"]:
            operand_id = instr.operand_ids[0]
            aliased_param = parameter_aliases.get(operand_id)
            if aliased_param is not None:
                parameter_aliases[instr.id] = aliased_param.new_tree(
                    instr, entry_comp.id
                )
        elif instr.opcode == "get-tuple-element":
            operand_id = instr.operand_ids[0]
            aliased_param = parameter_aliases.get(operand_id)
            if aliased_param is not None:
                aliased_param_idx = aliased_param.get_index(instr.tuple_index)
                if aliased_param_idx is not None:
                    parameter_aliases[instr.id] = aliased_param_idx.new_tree(
                        instr, entry_comp.id
                    )
        elif instr.opcode == "tuple":
            new_aliases = []
            have_alias = False
            for operand_id in instr.operand_ids:
                alias = parameter_aliases.get(operand_id)
                if alias is not None:
                    alias = alias.new_tree(instr, entry_comp.id)
                    have_alias = True
                new_aliases.append(alias)
            if have_alias:
                parameter_aliases[instr.id] = ParameterAlias(
                    aliases=new_aliases
                )
        elif instr.opcode == "call":
            _add_subcomp_aliases(
                instr, instr.called_computation_ids[0], instr.operand_ids
            )
        elif instr.opcode == "while":
            _add_subcomp_aliases(
                instr, instr.called_computation_ids[0], (instr.operand_ids[0],)
            )
        elif instr.opcode == "conditional":
            _add_subcomp_aliases(
                instr, instr.called_computation_ids[0], (instr.operand_ids[1],)
            )
            _add_subcomp_aliases(
                instr, instr.called_computation_ids[1], (instr.operand_ids[2],)
            )

    return parameter_aliases


def color_instructions(
    entry_comp: hlo_pb2.HloComputationProto,
    computation_map: dict[int, hlo_pb2.HloComputationProto],
    all_instructions: Mapping[int, hlo_pb2.HloInstructionProto],
    instruction_colors: Optional[dict[int, InstructionColoring]] = None,
    tuple_aliases: Optional[dict[int, int]] = None,
    color_generation: Optional[dict[LegateKey, int]] = None,
):
    instruction_colors = (
        {} if instruction_colors is None else instruction_colors
    )
    tuple_aliases = {} if tuple_aliases is None else tuple_aliases

    color_generation = {} if color_generation is None else color_generation
    next_generation = 0

    def color_tuple_arg_comp(
        input_tuple_id: int, comp_id: int
    ) -> InstructionColoring:
        tuple_coloring = instruction_colors[input_tuple_id]
        comp = computation_map[comp_id]

        tuple_arg = get_entry_tuple_arg(comp)
        instruction_colors[tuple_arg.id] = tuple_coloring
        tuple_aliases[tuple_arg.id] = input_tuple_id
        color_instructions(
            comp,
            computation_map,
            all_instructions,
            instruction_colors,
            tuple_aliases,
            color_generation,
        )
        tuple_arg_coloring = instruction_colors[tuple_arg.id]
        input_tuple = all_instructions[input_tuple_id]
        for operand_id, color in zip(
            input_tuple.operand_ids, tuple_arg_coloring.tuple_colors
        ):
            operand = all_instructions[operand_id]
            color_backwards(
                operand, color, all_instructions, instruction_colors
            )
        return instruction_colors[comp.root_id], comp

    for instr in entry_comp.instructions:
        color = instruction_colors.get(instr.id)
        if color is not None:
            # if explicitly give a color ahead of time,
            # just use that
            continue

        if "implicit_decomp" in instr.metadata.op_name:
            key = None
        else:
            key = LegateKey.create(instr.metadata.op_name)

        def _color_from_operands(instr):
            max_color = None
            max_generation = -1
            for operand_id in instr.operand_ids:
                operand_color = instruction_colors[operand_id]
                if operand_color is None:
                    raise Exception(
                        f"color None for operand {operand_id} on {instr.name}"
                    )
                if operand_color.explicit:
                    operand_color_generation = color_generation[
                        operand_color.explicit
                    ]
                    if operand_color_generation > max_generation:
                        max_color = operand_color.explicit
                        max_generation = operand_color_generation
            if max_color is None:
                color = InstructionColoring()
                instruction_colors[instr.id] = color
            else:
                color = InstructionColoring(color=max_color)
                color_backwards(
                    instr, color, all_instructions, instruction_colors
                )
            return color

        if key is None:
            if instr.opcode == "while":
                color, _ = color_tuple_arg_comp(
                    instr.operand_ids[0], instr.called_computation_ids[0]
                )
                instruction_colors[instr.id] = color.copy()
            elif instr.opcode == "conditional":
                false_colors, false_comp = color_tuple_arg_comp(
                    instr.operand_ids[2], instr.called_computation_ids[1]
                )
                true_colors, true_comp = color_tuple_arg_comp(
                    instr.operand_ids[1], instr.called_computation_ids[0]
                )
                merged_colors = []
                # Constants or other weirdness can cause false comp and
                # true comp to have different coloring for the output tuple.
                # This is not allowed since the tuples would have different
                # shapes after color partitioning. We therefore have to merge
                # the colors then propagate backwards into the computations
                # to force color agreement.
                for idx, (true_color, false_color) in enumerate(
                    zip(true_colors.tuple_colors, false_colors.tuple_colors)
                ):
                    merged_color, _ = true_color.merge(false_color)
                    if merged_color is None:
                        raise Exception(
                            f"{instr.name}: branches have incompatible "
                            f"coloring on index={idx}: "
                            f"{true_color} != {false_color}, "
                            f"{true_color == false_color}"
                        )
                    merged_colors.append(merged_color)
                merged_color = InstructionColoring(tuple_colors=merged_colors)

                color_backwards(
                    all_instructions[false_comp.root_id],
                    merged_color,
                    all_instructions,
                    instruction_colors,
                )
                color_backwards(
                    all_instructions[true_comp.root_id],
                    merged_color,
                    all_instructions,
                    instruction_colors,
                )

                instruction_colors[instr.id] = merged_color
                # the predicate variable needs to be colored with everything
                predicate = all_instructions[instr.operand_ids[0]]
                instruction_colors[predicate.id] = merged_color.copy()
                color_backwards(
                    predicate,
                    true_colors,
                    all_instructions,
                    instruction_colors,
                )
            elif instr.opcode == "tuple":
                tuple_colors = [
                    instruction_colors[operand_id]
                    for operand_id in instr.operand_ids
                ]
                instruction_colors[instr.id] = InstructionColoring(
                    tuple_colors=tuple_colors
                )
            elif instr.opcode == "call":
                comp = computation_map[instr.called_computation_ids[0]]

                # we can get weird scenarios in which the computation
                # has a unique color based on operands but then the
                # instructions inside the computation have a different color
                single_color = _color_from_operands(instr)
                if single_color.explicit:
                    for subinstr in comp.instructions:
                        instruction_colors[subinstr.id] = single_color.copy()
                    if is_tuple_shape(instr):
                        tuple_colors = [single_color.copy()] * len(
                            instr.shape.tuple_shapes
                        )
                        call_color = InstructionColoring(
                            tuple_colors=tuple_colors
                        )
                    else:
                        call_color = single_color.copy()
                else:
                    param_colors = [
                        instruction_colors[id] for id in instr.operand_ids
                    ]
                    params = [
                        instr
                        for instr in comp.instructions
                        if instr.opcode == "parameter"
                    ]
                    for comp_param, param_color, operand_id in zip(
                        params, param_colors, instr.operand_ids
                    ):
                        if param_color.explicit:
                            instruction_colors[
                                comp_param.id
                            ] = param_color.copy()

                    color_instructions(
                        comp,
                        computation_map,
                        all_instructions,
                        instruction_colors,
                        tuple_aliases,
                        color_generation,
                    )
                    call_color = instruction_colors[comp.root_id].copy()

                instruction_colors[instr.id] = call_color
                color_backwards(
                    instr, call_color, all_instructions, instruction_colors
                )
            elif instr.opcode == "parameter":
                if (
                    instr.shape.element_type
                    == xla_data_pb2.PrimitiveType.TUPLE
                ):
                    if instr.id in tuple_aliases:
                        alias_colors = instruction_colors[
                            tuple_aliases[instr.id]
                        ]
                        instruction_colors[instr.id] = InstructionColoring(
                            tuple_colors=alias_colors.tuple_colors
                        )
                    else:
                        tuple_colors = [
                            InstructionColoring()
                            for i in range(len(instr.shape.tuple_shapes))
                        ]
                        instruction_colors[instr.id] = InstructionColoring(
                            tuple_colors=tuple_colors
                        )
                else:
                    instruction_colors[instr.id] = InstructionColoring()
            elif instr.opcode == "get-tuple-element":
                tuple_coloring = instruction_colors[instr.operand_ids[0]]
                if tuple_coloring.tuple_colors is None:
                    raise Exception(
                        f"{instr.name} getting from {instr.operand_ids[0]} "
                        "does not have tuple colors"
                    )
                if instr.tuple_index >= len(tuple_coloring.tuple_colors):
                    raise Exception(
                        f"{instr.name} invalid index {instr.tuple_index}"
                        f"for tuple colors of length "
                        f"{len(tuple_coloring.tuple_colors)}"
                    )
                color = tuple_coloring.tuple_colors[instr.tuple_index].copy()
                instruction_colors[instr.id] = color
            else:
                color = _color_from_operands(instr)
                instruction_colors[instr.id] = color

        else:
            if key not in color_generation:
                color_generation[key] = next_generation
                next_generation += 1

            if instr.opcode == "tuple":
                color = InstructionColoring(
                    tuple_colors=[
                        InstructionColoring(color=key)
                        for i in range(len(instr.operand_ids))
                    ]
                )
            elif instr.opcode == "opt-barrier" and is_tuple_shape(instr):
                tuple_colors = [
                    InstructionColoring(color=key)
                    for _ in instr.shape.tuple_shapes
                ]
                color = InstructionColoring(tuple_colors=tuple_colors)
            else:
                color = InstructionColoring(color=key)

            instruction_colors[instr.id] = color
            color_backwards(instr, color, all_instructions, instruction_colors)

    return instruction_colors, tuple_aliases


class LayerSet:
    def __init__(self, computation_map):
        self.layer_sequential_order = []
        self.layers = {}
        self.layer_numbers = {}
        self.computation_map = computation_map

    def __iter__(self):
        return iter(self.layer_sequential_order)

    def get_layer(self, key: LegateKey) -> HloLayer:
        if key not in self.layers:
            layer = HloLayer(key, self.computation_map)
            self.layers[key] = layer
            self.layer_sequential_order.append(layer)
            self.layer_numbers[key] = len(self.layer_sequential_order)
            return layer
        return self.layers[key]

    def add_rematerialization(
        self,
        comp: hlo_pb2.HloComputationProto,
        instr: hlo_pb2.HloInstructionProto,
        key: LegateKey,
    ):
        self.get_layer(key).add_rematerialization(comp.id, instr)

    def add_instruction(
        self,
        comp: hlo_pb2.HloComputationProto,
        instr: hlo_pb2.HloInstructionProto,
        key: LegateKey,
        all_instructions: Optional[
            Mapping[int, hlo_pb2.HloInstructionProto]
        ] = None,
    ):
        self.get_layer(key).add_instruction(comp.id, instr, all_instructions)


def decompose_into_colors(
    entry_comp: hlo_pb2.HloComputationProto,
    computation_map: dict[int, hlo_pb2.HloComputationProto],
    instruction_colors: dict[int, InstructionColoring],
    tuple_aliases: dict[int, int],
    param_map: Mapping[int, hlo_pb2.HloInstructionProto],
    layer_set: Optional[LayerSet] = None,
    include_root: bool = True,
) -> List[HloLayer]:
    layer_set = LayerSet(computation_map) if layer_set is None else layer_set

    all_instructions: dict[int, hlo_pb2.HloInstructionProto] = {}
    for instr in entry_comp.instructions:
        if instr.id == entry_comp.root_id and not include_root:
            # do not add the root to any layers
            continue

        coloring = instruction_colors[instr.id]
        all_instructions[instr.id] = instr
        if instr.opcode == "while":
            body_comp = computation_map[instr.called_computation_ids[0]]
            decompose_into_colors(
                body_comp,
                computation_map,
                instruction_colors,
                tuple_aliases,
                param_map,
                layer_set,
            )
        elif instr.opcode == "conditional":
            true_comp = computation_map[instr.called_computation_ids[0]]
            false_comp = computation_map[instr.called_computation_ids[1]]
            decompose_into_colors(
                true_comp,
                computation_map,
                instruction_colors,
                tuple_aliases,
                param_map,
                layer_set,
            )
            decompose_into_colors(
                false_comp,
                computation_map,
                instruction_colors,
                tuple_aliases,
                param_map,
                layer_set,
            )
        elif instr.opcode == "call":
            comp = computation_map[instr.called_computation_ids[0]]
            decompose_into_colors(
                comp,
                computation_map,
                instruction_colors,
                tuple_aliases,
                param_map,
                layer_set,
            )

        run_config = RunConfig()
        for color in coloring.all_colors():
            layer_set.add_instruction(
                entry_comp, instr, color, all_instructions
            )
            if run_config.rematerialization and not color.is_backward:
                # the first instruction in every layer should NOT be recomputed
                # otherwise you will get thrashing in the recomputation or
                # you will put every activation in memory again and lose the
                # whole advantage of rematerialization
                if (
                    "activation_checkpoint" not in instr.metadata.op_name
                    or instr.id in param_map
                ):
                    # copy all forward instructions to the backprop
                    bkwd_color = color.replace(is_backward=True)
                    layer_set.add_rematerialization(
                        entry_comp, instr, bkwd_color
                    )

    return layer_set


class HloModule:
    def __init__(
        self,
        key: LegateKey,
        parameters: list[hlo_pb2.HloInstructionProto],
        inputs: list[hlo_pb2.HloInstructionProto],
        microbatch_inputs: list[hlo_pb2.HloInstructionProto],
        roots: list[hlo_pb2.HloInstructionProto],
        outputs: list[hlo_pb2.HloInstructionProto],
        microbatch_outputs: list[hlo_pb2.HloInstructionProto],
        hlo_module: Optional[hlo_pb2.HloModuleProto],
        num_microbatches: Optional[int] = None,
    ):
        self.hlo_module = hlo_module
        self.name = key.name if hlo_module is None else hlo_module.name
        self.key = key
        self.parameters = parameters
        self.inputs = inputs
        self.microbatch_inputs = microbatch_inputs
        self.roots = roots
        self.outputs = outputs
        self.microbatch_outputs = microbatch_outputs
        self._default_module_loaded = False
        self._sharded_modules_loaded: Mapping[TaskMesh, ShardedHloModule] = {}

        self.num_microbatches = num_microbatches

        # validate that all "interface" variables (inputs/outputs)
        # have legate_axes annotations
        validate_logical_shardings = False
        if validate_logical_shardings:
            for hlo_value in self.inputs + self.outputs:
                ndim = len(hlo_value.shape.dimensions)
                size = (
                    np.asarray(hlo_value.shape.dimensions).prod()
                    if ndim >= 1
                    else 1
                )
                axes = TaskMesh.get_legate_axes(hlo_value)

                if is_replicated_instruction(hlo_value):
                    sys.stderr.write(
                        "%s %s does not need logical axes\n"
                        % (hlo_value.name, hlo_value.shape.dimensions)
                    )
                elif axes is None:
                    sys.stderr.write(
                        "%s %s does not have logical axes\n"
                        % (hlo_value.name, hlo_value.shape.dimensions)
                    )
                else:
                    if all(ax is None for ax in axes):
                        axis_type = "no"
                    else:
                        axis_type = "all"
                        for ax, size in zip(axes, hlo_value.shape.dimensions):
                            if ax is None and size > 16:
                                axis_type = "partial"
                    sys.stderr.write(
                        f"{hlo_value.name} {hlo_value.shape.dimensions} has "
                        f"{axis_type} axes: {axes}\n"
                    )

        self._loaded = False
        self._default_hlo_id = -1

        self.submodules: list[HloModule] = []

    @property
    def interface_inputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.inputs + self.microbatch_inputs

    @property
    def interface_outputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.outputs + self.microbatch_outputs

    @property
    def batch_inputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.parameters + self.inputs

    @property
    def batch_outputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.roots + self.outputs

    @property
    def all_inputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.parameters + self.inputs + self.microbatch_inputs

    @property
    def all_outputs(self) -> list[hlo_pb2.HloInstructionProto]:
        return self.roots + self.outputs + self.microbatch_outputs

    def create_tensors(
        self,
        match_inputs: bool = False,
        tensor_map: Optional[TensorMap] = None,
    ) -> TensorMap:
        if tensor_map is None:
            tensor_map = TensorMap()

        if self.submodules:
            for module in self.submodules:
                module.create_tensors(tensor_map=tensor_map)
        else:
            for param in self.parameters:
                tensor_map.get_parameter(param)

            if match_inputs:
                self.match_inputs_and_outputs(tensor_map)

            for root in self.roots:
                tensor_map.get_root(root)

        return tensor_map

    # The layer name is unique to an HloLayer object.
    # The key is a logical identifier that can be shared by multiple layers,
    # e.g. layer0.forward and layer0.backward
    @property
    def task_name(self) -> str:
        return LegateKey.prefix(self._name)

    def leaves(self) -> List[HloModule]:
        if self.submodules:
            leaves = []
            for module in self.submodules:
                leaves.extend(module.leaves())
            return leaves
        return [self]

    def summary(self, mesh: Optional[GlobalMesh] = None):
        return self._module_string(indent="", mesh=mesh)

    def __call__(
        self,
        tensor_map: TensorMap,
        global_mesh: Optional[GlobalMesh] = None,
        init_tensors: bool = False,
        dry_run: bool = False,
    ) -> None:
        if not self._loaded:
            self.load(global_mesh)
        self._launch_module(tensor_map, global_mesh, init_tensors, dry_run)

    def load(
        self, global_mesh: Optional[GlobalMesh] = None, debug: bool = False
    ) -> None:
        self._load_module(global_mesh, debug)
        runtime.issue_execution_fence()

    def _load_module(
        self, global_mesh: Optional[GlobalMesh] = None, debug: bool = False
    ) -> None:
        if self._loaded:
            return
        self._loaded = True

        if self.submodules:
            for submodule in self.submodules:
                submodule._load_module(global_mesh=global_mesh, debug=debug)
                runtime.issue_execution_fence(block=True)
        else:
            if self.hlo_module is None:
                raise Exception(f"HloModule {self.name} has no HloModuleProto")

            # sharding annotations has to be the absolute last thing we do
            # when the module is loaded. The module might be split further
            # or have its device mesh modified.  Only when we finally commit
            # to loading the module should the logical mesh axes be converted
            # physical device shardings
            task_mesh = self._get_task_mesh(global_mesh)
            if task_mesh is not None:
                sharded_module = self._sharded_modules_loaded.get(task_mesh)
                if sharded_module is not None:
                    return  # already loaded, can return
                else:
                    sharded_module = compute_module_sharding(
                        self.hlo_module,
                        self.all_inputs,
                        self.all_outputs,
                        task_mesh,
                    )
                    self._sharded_modules_loaded[task_mesh] = sharded_module
                    hlo_str = str(sharded_module.hlo_module)
                    sharded_module.id = runtime.load_hlo(
                        hlo_str, self.name, task_mesh, debug
                    )
            else:
                if self._default_module_loaded:
                    return
                hlo_str = str(self.hlo_module)
                self._default_module_loaded = True
                self._default_hlo_id = runtime.load_hlo(hlo_str, self.name)

    def _get_task_mesh(
        self, global_mesh: Optional[GlobalMesh]
    ) -> Optional[TaskMesh]:
        if global_mesh is not None:
            task_mesh = global_mesh.get_task_mesh(self.name)
            if task_mesh is None:
                error_msg = (
                    "module {} has no task mesh "
                    "defined in global mesh\n{}".format(
                        self.name, str(global_mesh)
                    )
                )
                raise Exception(error_msg)
            return task_mesh
        return None

    @staticmethod
    def _create_partitioned_stores(
        tensors: Sequence[Tensor],
        shardings: Mapping[int, xla_data_pb2.OpSharding],
        task_mesh: TaskMesh,
        run_config: RunConfig,
    ) -> List[Union[Store, StorePartition]]:
        partitioned_stores = []

        def _is_scalar(shape):
            for dim in shape:
                if dim > 1:
                    return False
            return True

        for tensor in tensors:
            sharding = shardings.get(tensor.hlo_id)
            if sharding is None:
                if len(tensor.shape) >= 4:
                    raise Exception(
                        f"{tensor.name} id={tensor.hlo_id} is replicated,"
                        f"but should be partitioned: {list(shardings.keys())}"
                    )
                partitioned_stores.append(tensor.replicate(task_mesh.ndevices))
            else:
                partitioned_stores.append(tensor.partition_by_tiling(sharding))
        return partitioned_stores

    def _launch_module(
        self,
        tensor_map: TensorMap,
        global_mesh: Optional[GlobalMesh] = None,
        init_tensors: bool = False,
        dry_run: bool = False,
    ) -> None:
        run_config = RunConfig()

        self._load_module(global_mesh=global_mesh, debug=dry_run)
        if self.submodules:
            for submodule in self.submodules:
                submodule(tensor_map, global_mesh, init_tensors, dry_run)
            return

        task_mesh = self._get_task_mesh(global_mesh)

        microbatches = (
            [None]
            if self.num_microbatches is None
            else range(self.num_microbatches)
        )

        params = [
            tensor_map.get_parameter(param) for param in self.batch_inputs
        ]
        roots = [tensor_map.get_root(root) for root in self.batch_outputs]
        for mb in microbatches:
            runtime.legate_runtime.set_provenance(f"{self.name}.mb.{mb}")
            inputs = params + [
                tensor_map.get_input(input, mb)
                for input in self.microbatch_inputs
            ]
            outputs = roots + [
                tensor_map.get_output(output, mb)
                for output in self.microbatch_outputs
            ]
            if task_mesh is None:
                hlo_id = self._default_hlo_id
                input_stores = [tensor.store for tensor in inputs]
                output_stores = [tensor.store for tensor in outputs]
                if hlo_id == -1:
                    raise Exception(
                        f"{self.name} produced default module with id=-1"
                    )
                output_red_ops = [
                    ReductionOp.MAX if is_scalar(output) else None
                    for output in self.roots + self.outputs
                ]

            else:
                sharded_module = self._sharded_modules_loaded[task_mesh]
                hlo_id = sharded_module.id
                input_stores = self._create_partitioned_stores(
                    inputs,
                    sharded_module.input_shardings,
                    task_mesh,
                    run_config,
                )

                # force a sanity check on the size of the inputs
                if not run_config.explicit_replication:
                    for inp in self.interface_inputs:
                        is_replicated_instruction(
                            inp, sharded_module.input_shardings.get(inp.id)
                        )

                output_stores = self._create_partitioned_stores(
                    outputs,
                    sharded_module.output_shardings,
                    task_mesh,
                    run_config,
                )
                if hlo_id == -1:
                    raise Exception(
                        f"{self.name} produced sharded module with id=-1"
                    )

                output_red_ops = [
                    ReductionOp.MAX if isinstance(output, Store) else None
                    for output in output_stores
                ]

            if not dry_run:
                runtime.execute_hlo(
                    hlo_id,
                    self.name,
                    input_stores,
                    output_stores,
                    output_red_ops,
                    task_mesh,
                    init_tensors,
                )
            runtime.legate_runtime.reset_provenance()

    def print_decomposition_tree(self, indent: str = "") -> None:
        if self.submodules:
            print(indent + f"Module {self.name} is decomposed into:")
            for submodule in self.submodules:
                submodule.print_decomposition_tree(indent + "  ")
        else:
            print(indent + f"Module {self.name}")

    def decompose_into_layers(self, tensors: TensorMap) -> None:
        computation_map: dict[int, hlo_pb2.HloComputationProto] = dict(
            (comp.id, comp) for comp in self.hlo_module.computations
        )

        entry_comp = computation_map[self.hlo_module.entry_computation_id]
        entry_def_map = dict(
            (instr.id, instr) for instr in entry_comp.instructions
        )

        all_instructions: dict[int, hlo_pb2.HloInstructionProto] = dict(
            (instr.id, instr)
            for comp in self.hlo_module.computations
            for instr in comp.instructions
        )

        params = get_parameters(entry_comp)
        param_map = dict((param.id, param) for param in params)
        roots, _, _ = get_roots(
            entry_comp, entry_def_map, prune_constants=False
        )

        # The backwards passes may operate in reshape/convert
        # of the input parameters rather than the parameters themselves.
        # We need to de-alias the parameters here.
        parameter_aliases = compute_parameter_aliases(
            entry_comp, computation_map, all_instructions
        )

        instruction_colors, tuple_aliases = color_instructions(
            entry_comp, computation_map, all_instructions
        )

        # loop all the instructions and propagate parameter aliases
        # if any instructions are operating on parameter aliases
        # we need to propagate the color backwards so the parameter
        # ends up in the required layer
        for comp in self.hlo_module.computations:
            for instr in comp.instructions:
                for idx, operand_id in enumerate(instr.operand_ids):
                    if operand_id in parameter_aliases:
                        param_alias = parameter_aliases[operand_id]
                        if param_alias.tree is not None:
                            instr_color = instruction_colors[instr.id]
                            if instr_color.tuple_colors:
                                instr_color = instr_color.tuple_colors[idx]
                            for tree_instr, _ in param_alias.tree:
                                if not is_tuple_shape(tree_instr):
                                    alias_color = instruction_colors[
                                        tree_instr.id
                                    ]
                                    alias_color.append(instr_color)

        layers = decompose_into_colors(
            entry_comp,
            computation_map,
            instruction_colors,
            tuple_aliases,
            param_map,
            include_root=False,
        )

        microbatch_layers: list[HloLayer] = []
        for layer in layers:
            layer.compress_tuples()
            split_layers = layer.split_microbatches()
            microbatch_layers.extend(split_layers)

        inputs_needed = set()
        for layer in microbatch_layers:
            inputs_needed = inputs_needed.union(
                layer.compute_inputs(all_instructions, param_map)
            )

        all_outputs = inputs_needed.copy()
        all_roots = set(root.id for root in roots)
        for layer in microbatch_layers:
            layer.compute_outputs(all_outputs, all_roots)
            print(layer)

        queue = HloLayerQueue(microbatch_layers)
        sorted_layers: list[HloLayer] = []
        cleared_layers = []
        next = queue.pop_ready()
        outputs_produced = set()
        while next is not None:
            next.recompute_outputs(outputs_produced)
            # only include this layer if it actually still
            # produces useful output after prunining
            if len(next.all_outputs) > 0:
                sorted_layers.append(next)
                queue.finish(next)
            cleared_layers.append(next)
            next = queue.pop_ready()

        if len(cleared_layers) != len(microbatch_layers):
            for instr, layers in queue.depends_on.items():
                for layer in layers:
                    print(f"{layer.key} waiting on {instr}")
            raise Exception("not all layers are ready")

        for layer in sorted_layers:
            self.submodules.append(layer.to_module(tensors))

        if not self.roots:
            raise Exception("Module {} has no outputs".format(self.name))

        # we do not require all the input instructions to be in the map
        all_outputs = set(output.id for output in self.outputs)
        sub_outputs = set(
            root.id for layer in sorted_layers for root in layer.roots
        )
        missing_outputs = all_outputs - sub_outputs
        if missing_outputs:
            for instr_id in missing_outputs:
                print(f"Instr {instr_id} did not get assigned anywhere: ")
            raise Exception(
                f"Some root output instructions from the parent "
                f"{self.key} are missing from the submodules"
            )

    def backprop_label_propagation(
        hlo_module: hlo_pb2.HloModuleProto,
    ) -> HloModule:
        all_instructions = {}
        all_computations = {}
        return_tuples = {}

        sub_computation_types = ["while", "conditional"]

        for comp in hlo_module.computations:
            all_computations[comp.id] = comp
            if comp.id == hlo_module.entry_computation_id:
                entry_comp = comp
            for instr in comp.instructions:
                all_instructions[instr.id] = instr
                if instr.opcode == "while":
                    comp = all_computations[instr.called_computation_ids[0]]
                    root = all_instructions[comp.root_id]
                    input_tuple = all_instructions[instr.operand_ids[0]]
                    return_tuples[instr.id] = [(root, input_tuple, comp)]
                elif instr.opcode == "conditional":
                    true_comp = all_computations[
                        instr.called_computation_ids[0]
                    ]
                    true_root = all_instructions[true_comp.root_id]
                    true_input = all_instructions[instr.operand_ids[1]]

                    false_comp = all_computations[
                        instr.called_computation_ids[1]
                    ]
                    false_root = all_instructions[true_comp.root_id]
                    false_input = all_instructions[instr.operand_ids[2]]
                    return_tuples[instr.id] = [
                        (true_root, true_input, true_comp),
                        (false_root, false_input, false_comp),
                    ]

        gradient_scalar_instr = []
        for comp in hlo_module.computations:
            for instr in comp.instructions:
                if (
                    "legate_gradient_scalar" in instr.metadata.op_name
                    and "transpose(jvp" not in instr.metadata.op_name
                ):
                    gradient_scalar_instr.append(instr)

        # there is no marked gradient scalar, return the original module
        if not gradient_scalar_instr:
            for instr in entry_comp.instructions:
                if "transpose(jvp" in instr.metadata.op_name:
                    raise Exception(
                        "Backprop found in module, "
                        "but no scalar is marked as the loss function"
                    )
            return hlo_module

        scalar_def_use_tree = OrderedSet()

        def visit_sub_instruction(instr, comp):
            params_to_visit = []
            to_visit = [instr]

            while to_visit:
                instr = to_visit.pop()
                if instr.id not in scalar_def_use_tree:
                    scalar_def_use_tree.add(instr.id)
                    if instr.opcode == "get-tuple-element":
                        tuple_parent_id = instr.operand_ids[0]
                        tuple_parent = all_instructions[tuple_parent_id]
                        if tuple_parent_id in return_tuples:
                            for (
                                root_tuple,
                                input_tuple,
                                sub_comp,
                            ) in return_tuples[tuple_parent_id]:
                                # this aliases a root instruction in subcomp
                                # work backwards from the relevant instruction
                                # in the subcomputation
                                root_instr_id = root_tuple.operand_ids[
                                    instr.tuple_index
                                ]
                                root_instr = all_instructions[root_instr_id]
                                sub_params = visit_sub_instruction(
                                    root_instr, sub_comp
                                )
                                # this might propagate backwards to some inputs
                                # of the subcomputation
                                for param in sub_params:
                                    if param.opcode == "parameter":
                                        number = param.parameter_number
                                    else:
                                        number = param.tuple_index
                                    parent_param_alias = all_instructions[
                                        input_tuple.operand_ids[number]
                                    ]
                                    to_visit.append(parent_param_alias)
                        elif tuple_parent.opcode == "parameter":
                            params_to_visit.append(instr)
                        else:
                            # add the parent tuple to the visit
                            to_visit.append(tuple_parent)
                    elif instr.opcode == "parameter":
                        params_to_visit.append(instr)
                    elif instr.opcode not in sub_computation_types:
                        for operand_id in instr.operand_ids:
                            to_visit.append(all_instructions[operand_id])

                # do not descend into while loops automatically
                if instr.opcode not in sub_computation_types:
                    for comp_id in instr.called_computation_ids:
                        comp = all_computations[comp_id]
                        for instr in comp.instructions:
                            to_visit.append(instr)

            return params_to_visit

        if gradient_scalar_instr:
            for instr in gradient_scalar_instr:
                visit_sub_instruction(instr, entry_comp)

        marked_module = hlo_pb2.HloModuleProto()
        marked_module.CopyFrom(hlo_module)
        del marked_module.computations[:]
        for comp in hlo_module.computations:
            marked_comp = hlo_pb2.HloComputationProto()
            marked_comp.CopyFrom(comp)
            del marked_comp.instructions[:]
            for instr in comp.instructions:
                marked_instr = hlo_pb2.HloInstructionProto()
                marked_instr.CopyFrom(instr)
                if marked_instr.id in scalar_def_use_tree:
                    if "transpose(jvp" in marked_instr.metadata.op_name:
                        raise Exception(
                            f"why tf is {marked_instr.name} in the forward"
                        )
                    marked_instr.metadata.op_name += "/legate_forward/"
                marked_comp.instructions.append(marked_instr)
            marked_module.computations.append(marked_comp)
        return marked_module

    def custom_marking_propagation(
        hlo_module: hlo_pb2.HloModuleProto,
    ) -> HloModule:
        # first add all sharding annotations to the module
        # and get rid of all the custom call sharding ops
        marked_module = hlo_pb2.HloModuleProto()
        marked_module.CopyFrom(hlo_module)
        # clear the computations, we will rebuild them
        del marked_module.computations[:]

        all_instructions = {}
        all_sharded_instructions = {}

        for comp in hlo_module.computations:
            marked_comp = hlo_pb2.HloComputationProto()
            marked_comp.CopyFrom(comp)
            del marked_comp.instructions[:]
            marked_instructions = []
            inserted_instructions = {}
            custom_calls_replaced = {}

            def find_base_instruction(_id):
                while _id in custom_calls_replaced:
                    _id = custom_calls_replaced[_id]
                return _id

            for instr in comp.instructions:
                all_instructions[instr.id] = instr
                axes = TaskMesh.get_legate_axes(instr)
                # erase whatever sharding might exist in the input module
                # we will override it with our sharding annotations
                instr.ClearField("sharding")
                if (
                    instr.opcode == "custom-call"
                    and instr.custom_call_target == "Sharding"
                ):
                    base_instruction_id = find_base_instruction(
                        instr.operand_ids[0]
                    )
                    marked_instr = inserted_instructions[base_instruction_id]
                    if "activation_checkpoint" in instr.metadata.op_name:
                        marked_instr.metadata.op_name += (
                            "/activation_checkpoint/"
                        )

                    if axes is not None:
                        # Add a legate_axes spec to the op metadata.
                        # The legate_axes will be converted to an actual
                        # op sharding annotation later.
                        all_sharded_instructions[marked_instr.id] = axes
                        marked_instr.metadata.op_name += (
                            f"/legate_axes={axes}/"
                        )
                    # Do not append this sharding instruction,
                    # it gets removed and ignored.
                    custom_calls_replaced[instr.id] = marked_instr.id
                else:
                    marked_instr = hlo_pb2.HloInstructionProto()
                    marked_instr.CopyFrom(instr)
                    inserted_instructions[marked_instr.id] = marked_instr

                    # Remap all operands that were replaced
                    # custom calls to the call operand
                    del marked_instr.operand_ids[:]
                    for operand_id in instr.operand_ids:
                        if operand_id in custom_calls_replaced:
                            marked_instr.operand_ids.append(
                                find_base_instruction(operand_id)
                            )
                        else:
                            marked_instr.operand_ids.append(operand_id)
                    marked_instructions.append(marked_instr)
            if comp.root_id in custom_calls_replaced:
                # The root was a custom call that got removed,
                # make the operand the new ropt
                marked_comp.root_id = find_base_instruction(comp.root_id)
            for marked_instr in marked_instructions:
                # This append creates a copy, which means this must
                # come at the very end after all modifications and
                # annotations have been done
                marked_comp.instructions.append(marked_instr)
            marked_module.computations.append(marked_comp)

        all_computations = dict(
            (comp.id, comp) for comp in marked_module.computations
        )

        # now try to derive the axes for other ops that are not explicitly
        # marked with legate_axes annotations
        all_instructions = {}
        entry_comp = find_entry_computation(marked_module)
        known_axes = compute_sharding_propagation(entry_comp, all_computations)
        for comp in marked_module.computations:
            for instr in comp.instructions:
                # if this has no axis marking, but we were able to derive it
                # add the annotation to the instruction now
                if "legate_axes" not in instr.metadata.op_name:
                    axes = known_axes.get(instr.id)
                    if (
                        axes is not None and axes
                    ):  # don't put an empty tuple in
                        # known axes can be an arbitrary typing.Sequence
                        # we want to write these as a tuple
                        axes = tuple(axes)
                        # mark the axes here ax implicitly derived since we may
                        # not want to distinguish explicitly sharded from
                        # implicitly derived when choosing which instructions
                        # to add physical sharding annotations to
                        instr.metadata.op_name += (
                            f"/legate_axes={axes}/implicit_axes/"
                        )

        return marked_module

    def match_inputs_and_outputs(self, tensor_map: TensorMap) -> None:
        """Find a best matching between inputs and outputs

        Returns:
            Mapping[int,int]: Mapping of output IDs to the best input ID match.
                              Not all inputs or outputs will have matches.
        """
        parameters = set(param.id for param in self.parameters)
        roots = set(root.id for root in self.roots)
        entry_comp = find_entry_computation(self.hlo_module)

        # the set of all instructions that are connected to a parameter
        # through element-wise ops
        connected_instructions = set()
        users = {}

        all_parameter_distances = {param.id: {} for param in self.parameters}

        # build a graph where only instructions connected to parameters
        # through elementwise ops (i.e. shape preserving ops)
        # run bellman-ford on subgraph to determine the shortest distance
        # between all input/output pairs
        for instr in entry_comp.instructions:
            if instr.id in parameters:
                connected_instructions.add(instr.id)
                users[instr.id] = []
                all_parameter_distances[instr.id][instr.id] = 0
            elif instr.id in roots or instr.opcode in _SIZE_PRESERVING_OPS:
                for operand_id in instr.operand_ids:
                    # only add instructions that can be traced back to params
                    if operand_id in connected_instructions:
                        connected_instructions.add(instr.id)
                        if operand_id not in users:
                            users[operand_id] = []
                        users[operand_id].append(instr)
                        for param_id in parameters:
                            param_distances = all_parameter_distances[param_id]
                            operand_distance = param_distances.get(operand_id)
                            # if this operand is connected to the parameters
                            # see if it gives a shorter distance for the user
                            if operand_distance is not None:
                                min_distance = param_distances.get(
                                    instr.id, 10000
                                )
                                new_distance = operand_distance + 1
                                if new_distance < min_distance:
                                    param_distances[instr.id] = new_distance

        for param in self.parameters:
            param_distances = all_parameter_distances[param.id]
            min_distance = 100000
            min_match = None
            for root in self.roots:
                distance = param_distances.get(root.id, None)
                # if new shortest distance AND they have the same shape
                if (
                    distance is not None
                    and distance < min_distance
                    and param.shape.dimensions == root.shape.dimensions
                ):
                    min_distance = distance
                    min_match = root

            # hopefully an output is not used more than once
            # I could do a bipartite match to choose a globally
            # best set of matches, but that seems like a lot of work
            # when most cases will not have conflicting matches
            if min_match is not None:
                tensor_map.add_parameter_alias(min_match.id, param.id)

    @staticmethod
    def create(hlo_module: hlo_pb2.HloModuleProto) -> HloModule:
        # first label instructions that are backprop/forward as necessary
        hlo_module = HloModule.backprop_label_propagation(hlo_module)

        task_id_offsets: dict[str, int] = {}

        all_computations = dict(
            (comp.id, comp) for comp in hlo_module.computations
        )
        entry_comp = all_computations[hlo_module.entry_computation_id]

        # before re-mapping legate keys, make sure microbatch task ids are
        # normalized to start from zero. Somehow different layers end up
        # with different microbatch numbering (presumably from jit/tracing?)
        for comp in hlo_module.computations:
            for instr in comp.instructions:
                key = LegateKey.create(instr.metadata.op_name)
                if key is not None and key.task_id is not None:
                    offset = task_id_offsets.get(key.name)
                    if offset is None:
                        task_id_offsets[key.name] = key.task_id
                    else:
                        task_id_offsets[key.name] = min(key.task_id, offset)

        for comp in hlo_module.computations:
            for instr in comp.instructions:
                key = LegateKey.create(instr.metadata.op_name)
                if key is not None and key.task_id is not None:
                    offset = task_id_offsets.get(key.name, 0)
                    LegateKey.set_task_id(instr, key.task_id - offset)

        # The very first thing that needs to be done is to remove all
        # the sharding custom calls and propagate the annotations
        hlo_module = HloModule.custom_marking_propagation(hlo_module)
        # we need to re-find the entry computation from the new module
        entry_comp = find_entry_computation(hlo_module)
        entry_def_map: dict[int, hlo_pb2.HloInstructionProto] = dict(
            (instr.id, instr) for instr in entry_comp.instructions
        )

        all_operands = set()
        all_instructions = {}
        layer_forward_checkpoints = {}

        # set the last forward instruction in each layer
        # as the activation checkpoint
        def _recurse_tree(
            comp: hlo_pb2.HloComputationProto, instruction_time: int = 0
        ):
            for instr in comp.instructions:
                key = LegateKey.create(instr.metadata.op_name)
                instr.metadata.op_name += (
                    f"/instruction_time={instruction_time}/"
                )
                if (
                    key is not None and not key.is_backward
                ):  # and instr.opcode != "custom-call":
                    layer_forward_checkpoints[key] = instr
                for comp_id in instr.called_computation_ids:
                    instruction_time = _recurse_tree(
                        all_computations[comp_id], instruction_time
                    )
                instruction_time += 1
            return instruction_time

        _recurse_tree(entry_comp)

        # reverse the computations so they go in "chronological" order
        for comp in hlo_module.computations:
            for instr in comp.instructions:
                all_instructions[instr.id] = instr
                constant = find_derived_constant(instr, all_instructions)
                for operand_id in instr.operand_ids:
                    all_operands.add(operand_id)
                if constant is None:
                    _ = map_instruction_legate_key(instr)
                else:
                    # anything that is a simple derivation of a constant
                    # should be replicated on all layers that need it
                    clear_legate_key(instr)
                    instr.metadata.op_name += "/replicate/"

        roots, found_root_constants, output_reindex = get_roots(
            entry_comp, entry_def_map
        )

        parameters = [
            instr
            for instr in entry_comp.instructions
            if instr.opcode == "parameter"
        ]
        parameters.sort(key=lambda instr: instr.parameter_number)

        # keeps track of whether any parameters are unused and can be
        # pruned from the module to avoid runtime overhead
        prune_parameters = False
        # if any inputs get pruned, we will need to reindex them
        input_reindex = {}
        # we might have tupled args, in which case we need to unroll them
        # into many parameters and ignore the parameter tuple
        if len(parameters) == 1 and is_tuple_shape(parameters[0]):
            param_id = parameters[0].id
            parameters = []
            for instr in entry_comp.instructions:
                if (
                    instr.opcode == "get-tuple-element"
                    and instr.operand_ids[0] == param_id
                ):
                    if instr.id in all_operands:  # this parameter is used
                        input_reindex[instr.tuple_index] = len(parameters)
                        parameters.append(instr)
                    else:
                        # this parameter is never used, which means we should
                        # prune it and remove it from the input set
                        prune_parameters = True
            parameters.sort(key=lambda instr: instr.tuple_index)
        else:
            pruned_parameters = []
            for idx, param in enumerate(parameters):
                if param.id in all_operands:  # this param is used
                    input_reindex[idx] = len(pruned_parameters)
                    pruned_parameters.append(param)
                else:
                    # this parameter is never used, which means we should
                    # prune it and remove it from the input set
                    prune_parameters = True
            parameters = pruned_parameters

        if prune_parameters or found_root_constants:
            parameters, roots, hlo_module = HloModule.prune_module(
                hlo_module, parameters, input_reindex, roots, output_reindex
            )

        top_key = LegateKey(hlo_module.name)
        return HloModule(
            top_key,
            parameters=parameters,
            inputs=[],
            microbatch_inputs=[],
            roots=roots,
            outputs=[],
            microbatch_outputs=[],
            hlo_module=hlo_module,
        )

    @staticmethod
    def prune_module(
        hlo_module: hlo_pb2.HloInstructionProto,
        parameters: List[hlo_pb2.HloInstructionProto],
        input_reindex: Mapping[int, int],
        roots: List[hlo_pb2.HloInstructionProto],
        output_reindex: Mapping[int, int],
    ):
        pruned_module = hlo_pb2.HloModuleProto()
        pruned_module.CopyFrom(hlo_module)
        entry_comp = find_entry_computation(pruned_module)

        all_param_ids = set([param.id for param in parameters])

        new_entry_comp = hlo_pb2.HloComputationProto()
        new_entry_comp.CopyFrom(entry_comp)
        del new_entry_comp.instructions[:]

        # don't grab the tuple_param until after
        params = [
            instr
            for instr in entry_comp.instructions
            if instr.opcode == "parameter"
        ]
        tupled_args = len(params) == 1 and is_tuple_shape(params[0])
        tuple_param = params[0] if tupled_args else None
        # Use a value of -1, which will never be an instruction ID
        # to simply the logic of the code below
        tuple_param_id = tuple_param.id if tuple_param is not None else -1

        # do not include pruned parameters in the final module
        for instr in entry_comp.instructions:
            if instr.opcode == "parameter":
                if instr.id in all_param_ids or instr.id == tuple_param_id:
                    new_entry_comp.instructions.append(instr)
            elif (
                instr.opcode == "get-tuple-element"
                and tupled_args
                and instr.operand_ids[0] == tuple_param_id
            ):
                if instr.tuple_index in input_reindex:
                    new_entry_comp.instructions.append(instr)
            else:
                new_entry_comp.instructions.append(instr)

        # replace the original entry comp with the pruned one
        entry_comp.CopyFrom(new_entry_comp)
        # rebuild the params array from the new entry computation
        params = [
            instr
            for instr in entry_comp.instructions
            if instr.opcode == "parameter"
        ]
        tuple_param = params[0] if tupled_args else None

        entry_def_map: dict[int, hlo_pb2.HloInstructionProto] = dict(
            (instr.id, instr) for instr in entry_comp.instructions
        )
        root_instr = find_root_instruction(pruned_module)
        root_instr.ClearField("sharding")
        del root_instr.operand_ids[:]
        root_shape = xla_data_pb2.ShapeProto()
        root_shape.element_type = xla_data_pb2.PrimitiveType.TUPLE

        new_roots = []  # clear and rebuild from new module
        for root in roots:
            root_instr.operand_ids.append(root.id)
            root_shape.tuple_shapes.append(root.shape)
            new_roots.append(entry_def_map[root.id])
        root_instr.shape.CopyFrom(root_shape)

        new_parameters = []
        arg_shape = xla_data_pb2.ShapeProto()
        arg_shape.element_type = xla_data_pb2.PrimitiveType.TUPLE
        arg_shapes = [None] * len(parameters)
        for param in parameters:
            new_param = entry_def_map[param.id]
            if new_param.opcode == "get-tuple-element":
                new_param.tuple_index = input_reindex[param.tuple_index]
                arg_shapes[new_param.tuple_index] = param.shape
            new_parameters.append(new_param)

        if tupled_args:
            for shape in arg_shapes:
                if shape is None:
                    raise Exception(
                        "missing shape for get-tuple-element parameter"
                    )
                arg_shape.tuple_shapes.append(shape)

        new_input_output_alias = hlo_pb2.HloInputOutputAliasProto()
        for entry in pruned_module.input_output_alias.entries:
            # TODO: this assumes the shape index is {0}, {1} and that
            # we can translate to an integer by taking a[0]
            old_output_index = entry.output_shape_index[0]
            new_output_index = output_reindex.get(old_output_index)
            old_input_index = (
                entry.parameter_shape_index[0]
                if tupled_args
                else entry.parameter_number
            )
            new_input_index = input_reindex.get(old_input_index)
            if new_input_index is not None and new_output_index is not None:
                new_entry = hlo_pb2.HloInputOutputAliasProto.AliasEntryProto()
                new_entry.CopyFrom(entry)
                del new_entry.output_shape_index[:]
                new_entry.output_shape_index.append(new_output_index)
                if tupled_args:
                    del new_entry.parameter_shape_index[:]
                    new_entry.parameter_shape_index.append(new_input_index)
                else:
                    new_entry.parameter_number = new_input_index
                new_input_output_alias.entries.append(new_entry)

        if tupled_args:
            tuple_param.ClearField("sharding")
            tuple_param.shape.CopyFrom(arg_shape)
            del entry_comp.program_shape.parameters[:]
            entry_comp.program_shape.parameters.append(arg_shape)
            new_parameters.sort(key=lambda instr: instr.tuple_index)
        else:
            del entry_comp.program_shape.parameters[:]
            del entry_comp.program_shape.parameter_names[:]
            for idx, param in enumerate(new_parameters):
                param.parameter_number = idx
                entry_comp.program_shape.parameters.append(param.shape)
                entry_comp.program_shape.parameter_names.append(param.name)

        pruned_module.input_output_alias.CopyFrom(new_input_output_alias)
        entry_comp.program_shape.result.CopyFrom(root_shape)
        pruned_module.host_program_shape.CopyFrom(entry_comp.program_shape)

        return new_parameters, new_roots, pruned_module


def load_module(path: str) -> HloModule:
    pb = open(path, "rb").read()
    hlo_proto = hlo_pb2.HloProto()
    hlo_proto.ParseFromString(pb)
    hlo_module = hlo_proto.hlo_module
    return HloModule.create(hlo_module)
