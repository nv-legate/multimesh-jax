from typing import Mapping, Optional, Sequence, Tuple

import gin
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.xla.service import hlo_pb2

from .mesh import TaskMesh

_ELEMENTWISE_OPS = {
    "sine",
    "cosine",
    "add",
    "subtract",
    "multiply",
    "convert",
    "divide",
    "constant",
    "log",
    "power",
    "sqrt",
    "negate",
    "rsqrt",
    "exponential",
    "tanh",
    "select",
    "and",
    "or",
    "compare",
}

_FUSIBLE_OPS = {
    "sine",
    "cosine",
    # "add",
    # "subtract",
    # "multiply",
    # "convert",
    # "divide",
    "log",
    "power",
    "sqrt",
    "negate",
    "rsqrt",
    "exponential",
    "tanh",
    "reshape",
}


# These are all the ops that keep the same size as operands,
# although they may change the shape
_SIZE_PRESERVING_OPS = _ELEMENTWISE_OPS
_SIZE_PRESERVING_OPS.add("reshape")

# ops with no operands that generate new values
_GENERATE_OPS = {"parameter", "constant", "iota"}

_UNLABELED_OPCODES = {
    "broadcast",
    "call",
    "slice",
    "constant",
    "tuple",
    "get-tuple-element",
    "reshape",
    "parameter",
    "negate",
}

_REDUCE_OPS = [
    "reduce",
    "maximum",
]

_CONTRACT_OPS = [
    "dot",
]

_RESHAPE_OPS = frozenset(
    [
        "reshape",
        "broadcast",
        "transpose",
        "call",  # the call can do anything, assume it changes shape
    ]
    + _REDUCE_OPS
    + _CONTRACT_OPS
)

_GATHER_SCATTER_OPS = {
    "gather",
    "scatter",
    "get-tuple-element",
    "tuple",
}

_CUSTOM_CALL_ELEMENTWISE_OPS = {"Sharding"}

_CUSTOM_CALL_RESHAPE_OPS = {}


@gin.configurable
def check_replicated_instruction_exception(
    instr: hlo_pb2.HloInstructionProto,
    exempt_axes: Sequence[Tuple[str, ...]] = {},
    allow_all: bool = True,
) -> None:
    if allow_all:
        return
    axes = list(TaskMesh.get_legate_axes(instr))
    axes.sort(key=str)
    for exemption in exempt_axes:
        exemption = list(exemption)
        exemption.sort(key=str)
        if all(lax == rax for lax, rax in zip(axes, exemption)):
            return  # this exemption matches
    # no exemption matches
    raise Exception(
        f"{instr.name}{instr.shape.dimensions} axes={axes} "
        "too large to be replicated"
    )


@gin.configurable
def is_replicated_instruction(
    instr: hlo_pb2.HloInstructionProto,
    sharding: Optional[xla_data_pb2.OpSharding] = None,
    max_ndim: int = 2,
    max_dim_size: int = 32,
):
    # we have to pass both the instruction and the sharding in case we want to
    # evaluate this with a custom sharding
    if sharding is None:
        sharding = instr.sharding
    if len(sharding.tile_assignment_dimensions) == 0:
        is_small = len(instr.shape.dimensions) <= max_ndim and all(
            dim <= max_dim_size for dim in instr.shape.dimensions
        )
        if not is_small:
            check_replicated_instruction_exception(instr)
        return True

    return False


def is_tuple_shape(instr: hlo_pb2.HloInstructionProto):
    return instr.shape.element_type == xla_data_pb2.PrimitiveType.TUPLE


def is_scalar(instr: hlo_pb2.HloInstructionProto):
    return len(instr.shape.dimensions) == 0


def instr_size(instr: hlo_pb2.HloInstructionProto, type_size: int = 4):
    size = type_size
    for dim in instr.shape.dimensions:
        size *= dim
    return size


def hlo_instruction_str(
    instr: hlo_pb2.HloInstructionProto,
    metadata: bool = False,
    sharding: bool = False,
    axes: bool = False,
    legate_annotations: bool = False,
    instructions: Mapping[int, hlo_pb2.HloInstructionProto] = {},
) -> str:
    if instructions:
        operand_strs = [instructions[id].name for id in instr.operand_ids]
        operand_str = ",".join(operand_strs)
    else:
        operand_str = ""

    if sharding and instr.sharding.tile_assignment_dimensions:
        sharding_str = str(instr.sharding.tile_assignment_dimensions)
    else:
        sharding_str = ""

    dims_str = str(instr.shape.dimensions)

    metadata_str = instr.metadata.op_name if metadata else ""

    legate_axes = TaskMesh.get_legate_axes(instr)
    if axes and legate_axes is not None:
        axes_str = str(legate_axes)
    else:
        axes_str = ""

    legate_str = ""
    if legate_annotations:
        if "implicit_decomp" in instr.metadata.op_name:
            legate_str += " implicit"

        if "replicate/" in instr.metadata.op_name:
            legate_str += " replicate"

    return (
        f"{instr.name:25} {instr.opcode}({operand_str}) "
        f"{dims_str}{axes_str}{sharding_str} {legate_str} {metadata_str}"
    )


def find_entry_computation(
    hlo_module: hlo_pb2.HloModuleProto,
) -> hlo_pb2.HloComputationProto:
    for comp in hlo_module.computations:
        if comp.id == hlo_module.entry_computation_id:
            return comp
    assert False
    return hlo_pb2.HloComputationProto()


def find_root_instruction(
    hlo_module: hlo_pb2.HloModuleProto,
) -> hlo_pb2.HloInstructionProto:
    entry_comp = find_entry_computation(hlo_module)
    for instr in entry_comp.instructions:
        if instr.id == entry_comp.root_id:
            return instr
    assert False
    return hlo_pb2.HloInstructionProto()


def _fill_nones(shape, lhs_axes, rhs_axes):
    if shape.element_type == xla_data_pb2.PrimitiveType.TUPLE:
        for subshape, lhs, rhs in zip(shape.tuple_shapes, lhs_axes, rhs_axes):
            _fill_nones(subshape, lhs, rhs)
    else:
        for idx, (lhs, rhs) in enumerate(zip(lhs_axes, rhs_axes)):
            if lhs is None and rhs is not None:
                lhs_axes[idx] = rhs
            if rhs is None and lhs is not None:
                rhs_axes[idx] = lhs


def _tuple_nones(shape):
    nones = []
    if shape.element_type == xla_data_pb2.PrimitiveType.TUPLE:
        for subshape in shape.tuple_shapes:
            nones.append(_tuple_nones(subshape))
    else:
        for entry in shape.dimensions:
            nones.append(None)
    return nones


def _sharding_propagation_helper(
    comp: hlo_pb2.HloComputationProto,
    all_computations: Mapping[int, hlo_pb2.HloComputationProto],
    known_axes: Mapping[int, Sequence[Optional[str]]],
    is_entry_computation: bool = True,
):
    # now try to derive the axes for other ops that are not explicitly
    # marked with legate_axes annotations
    all_instructions = {}
    updated = True
    known_sizes = {}

    def verify_size(instr, instr_axes, context):
        for dim_size, ax in zip(instr.shape.dimensions, instr_axes):
            if ax is not None:
                prev_entry = known_sizes.get(ax)
                if prev_entry is None:
                    known_sizes[ax] = (dim_size, context)
                else:
                    prev_size, prev_instr = prev_entry
                    # if prev_size != dim_size:
                    #    sys.stderr.write(
                    #        f"Instruction {context.name} set new size "
                    #        f"{dim_size} for {ax}, previously assigned "
                    #        f"{prev_size} on {prev_instr.name}\n"
                    #    )

    for id, axes in known_axes.items():
        if None in axes and isinstance(axes, tuple):
            # make the entry mutable since it will needs its Nones filled in
            axes = list(axes)
            known_axes[id] = axes

    def _recurse_to_comp(
        instr: hlo_pb2.HloInstructionProto,
        comp_id: int,
        axes: Sequence[Optional[str]],
    ):
        body_comp = all_computations[comp_id]
        for body_instr in body_comp.instructions:
            if body_instr.opcode == "parameter":
                known_axes[body_instr.id] = axes[:]
                break

        _sharding_propagation_helper(body_comp, all_computations, known_axes)

        return known_axes[body_comp.root_id]

    while updated:
        updated = False
        # TODO: this can be made much more efficient than just looping
        # over and over again until no new axes are derived
        for instr in comp.instructions:
            all_instructions[instr.id] = instr
            instr_axes = known_axes.get(instr.id)

            if instr_axes is None:
                instr_axes = TaskMesh.get_legate_axes(instr)
            ndim = len(instr.shape.dimensions)
            if instr_axes is not None:
                if None in instr_axes:
                    # this needs to be mutable
                    instr_axes = list(instr_axes)
            else:
                instr_axes = _tuple_nones(instr.shape)

            for operand_id in instr.operand_ids:
                operand = all_instructions.get(operand_id)
                if operand is not None:
                    operand_axes = known_axes[operand_id]

            # don't know how to handle something whose output type is tuple,
            # but isn't a tuple instruction
            if (
                instr.shape.element_type == xla_data_pb2.PrimitiveType.TUPLE
                and instr.opcode != "tuple"
                and instr.opcode not in ["call", "while", "conditional"]
            ):
                known_axes[instr.id] = instr_axes
                continue

            if instr.opcode == "tuple":
                instr_axes = []
                for operand_id in instr.operand_ids:
                    instr_axes.append(known_axes[operand_id])
            elif instr.opcode == "get-tuple-element":
                operand_axes = known_axes[instr.operand_ids[0]][
                    instr.tuple_index
                ]
                for idx, (instr_ax, operand_ax) in enumerate(
                    zip(instr_axes, operand_axes)
                ):
                    if operand_ax is None and instr_ax is not None:
                        operand_axes[idx] = instr_ax
                    if instr_ax is None and operand_ax is not None:
                        instr_axes[idx] = operand_ax
            elif instr.opcode == "dot":
                lhs, rhs = (all_instructions[id] for id in instr.operand_ids)
                lhs_axes = known_axes[lhs.id]
                rhs_axes = known_axes[rhs.id]
                dims = instr.dot_dimension_numbers

                lhs_cxn_dims = list(dims.lhs_contracting_dimensions)
                lhs_batch_dims = list(dims.lhs_batch_dimensions)

                lhs_all_dims = set(range(len(lhs.shape.dimensions)))
                lhs_external_dims = list(
                    lhs_all_dims - set(lhs_batch_dims + lhs_cxn_dims)
                )
                rhs_cxn_dims = list(dims.rhs_contracting_dimensions)
                rhs_batch_dims = list(dims.rhs_batch_dimensions)

                rhs_all_dims = set(range(len(rhs.shape.dimensions)))
                rhs_external_dims = list(
                    rhs_all_dims - set(rhs_batch_dims + rhs_cxn_dims)
                )

                for i, (lhs_dim, rhs_dim) in enumerate(
                    zip(lhs_batch_dims, rhs_batch_dims)
                ):
                    instr_ax = instr_axes[i]
                    lhs_ax = lhs_axes[lhs_dim]
                    rhs_ax = rhs_axes[rhs_dim]
                    if instr_ax is None:
                        if lhs_ax is not None:
                            instr_axes[i] = lhs_ax
                        elif rhs_ax is not None:
                            instr_axes[i] = rhs_ax
                    if lhs_ax is None:
                        if instr_ax is not None:
                            lhs_axes[lhs_dim] = instr_ax
                        elif rhs_ax is not None:
                            lhs_axes[lhs_dim] = rhs_ax
                    if rhs_ax is None:
                        if instr_ax is not None:
                            rhs_axes[rhs_dim] = instr_ax
                        elif lhs_ax is not None:
                            rhs_axes[rhs_dim] = lhs_ax

                offset = len(lhs_batch_dims)
                for i, lhs_dim in enumerate(lhs_external_dims):
                    lhs_ax = lhs_axes[lhs_dim]
                    instr_ax = instr_axes[i + offset]
                    if instr_ax is None and lhs_ax is not None:
                        instr_axes[i + offset] = lhs_ax
                    if lhs_ax is None and instr_ax is not None:
                        lhs_axes[lhs_dim] = instr_ax

                offset += len(lhs_external_dims)
                for i, rhs_dim in enumerate(rhs_external_dims):
                    rhs_ax = rhs_axes[rhs_dim]
                    instr_ax = instr_axes[i + offset]
                    if instr_ax is None and rhs_ax is not None:
                        instr_axes[i + offset] = rhs_ax
                        updated = True
                    if rhs_ax is None and instr_ax is not None:
                        rhs_axes[rhs_dim] = instr_ax
                        updated = True

                for lhs_dim, rhs_dim in zip(lhs_cxn_dims, rhs_cxn_dims):
                    lhs_ax = lhs_axes[lhs_dim]
                    rhs_ax = rhs_axes[rhs_dim]
                    if lhs_ax is not None and rhs_ax is None:
                        rhs_axes[rhs_dim] = lhs_ax
                        updated = True

                    if rhs_ax is not None and lhs_ax is None:
                        lhs_axes[lhs_dim] = rhs_ax
                        updated = True

                verify_size(lhs, lhs_axes, instr)
                verify_size(rhs, rhs_axes, instr)
            # elif instr.opcode == "slice":
            #    operand_id = instr.operand_ids[0]
            #    operand_axes = known_axes[operand_id]
            #    operand = all_instructions[operand_id]
            #    print(instr)
            #    for i in range(ndim):
            #        instr_ax = instr_axes[i]
            #        operand_ax = operand_axes[i]
            #        if instr_ax is None and operand_ax is not None:
            #            instr_axes[i] = operand_ax
            #        elif operand_ax is None and instr_ax is not None:
            #            operand_axes[i] = instr_ax
            elif instr.opcode == "transpose":
                operand_id = instr.operand_ids[0]
                operand_axes = known_axes[operand_id]
                operand = all_instructions[operand_id]
                for i in range(ndim):
                    operand_dim = instr.dimensions[i]
                    operand_ax = operand_axes[operand_dim]
                    instr_ax = instr_axes[i]
                    if instr_ax is None and operand_ax is not None:
                        instr_axes[i] = operand_ax
                        updated = True
                    if operand_ax is None and instr_ax is not None:
                        operand_axes[operand_dim] = instr_ax
                        updated = True
                verify_size(operand, operand_axes, instr)

            elif instr.opcode == "broadcast":
                operand_id = instr.operand_ids[0]
                operand = all_instructions[operand_id]
                operand_axes = known_axes[operand_id]

                operand_dim = 0
                for dim in range(len(instr.shape.dimensions)):
                    if dim in instr.dimensions:
                        operand_ax = operand_axes[operand_dim]
                        instr_ax = instr_axes[dim]
                        if instr_ax is None and operand_ax is not None:
                            instr_axes[dim] = operand_ax
                            updated = True
                        if operand_ax is None and instr_ax is not None:
                            operand_axes[operand_dim] = instr_ax
                            updated = True
                        operand_dim += 1
                verify_size(operand, operand_axes, instr)

            elif instr.opcode == "all-reduce":
                instr_axes = known_axes[instr.operand_ids[0]]

            elif instr.opcode == "reduce":
                operand_id = instr.operand_ids[0]
                operand = all_instructions[operand_id]
                operand_axes = known_axes[operand_id]

                instr_dim = 0
                for operand_dim in range(len(operand.shape.dimensions)):
                    if (
                        operand_dim not in instr.dimensions
                    ):  # this is not reduced away
                        operand_ax = operand_axes[operand_dim]
                        instr_ax = instr_axes[instr_dim]
                        if operand_ax is None and instr_ax is not None:
                            operand_axes[operand_dim] = instr_ax
                            updated = True
                        if instr_ax is None and operand_ax is not None:
                            instr_axes[instr_dim] = operand_ax
                            updated = True
                        instr_dim += 1
                verify_size(operand, operand_axes, instr)

            elif instr.opcode == "reshape":
                operand_id = instr.operand_ids[0]
                operand_axes = known_axes[operand_id]
                operand = all_instructions[operand_id]
                if (
                    len(instr.shape.dimensions) > 0
                    and len(operand.shape.dimensions) > 0
                ):
                    total_size = 1
                    for size in instr.shape.dimensions:
                        total_size *= size
                    instr_start = 0
                    instr_stop = 1
                    instr_size = instr.shape.dimensions[0]
                    operand_start = 0
                    operand_stop = 1
                    operand_size = operand.shape.dimensions[0]

                    matching_ranges = []

                    while operand_size < total_size or instr_size < total_size:
                        # if these are aligned, match the axes
                        if operand_size == instr_size:
                            matching_ranges.append(
                                [
                                    (operand_start, operand_stop),
                                    (instr_start, instr_stop),
                                ]
                            )
                            # if operand_stop < len(operand.shape.dimensions):
                            operand_size *= operand.shape.dimensions[
                                operand_stop
                            ]
                            # if instr_stop < len(instr.shape.dimensions):
                            instr_size *= instr.shape.dimensions[instr_stop]
                            operand_start = operand_stop
                            instr_start = instr_stop
                            operand_stop += 1
                            instr_stop += 1
                        elif operand_size < instr_size:
                            operand_size *= operand.shape.dimensions[
                                operand_stop
                            ]
                            operand_stop += 1
                        elif instr_size < operand_size:
                            instr_size *= instr.shape.dimensions[instr_stop]
                            instr_stop += 1
                    # trailing match didn't get picked up
                    if instr_start < len(
                        instr.shape.dimensions
                    ) or operand_start < len(operand.shape.dimensions):
                        matching_ranges.append(
                            [
                                (operand_start, operand_stop),
                                (instr_start, instr_stop),
                            ]
                        )

                    for range_pair in matching_ranges:
                        (operand_start, operand_stop), (
                            instr_start,
                            instr_stop,
                        ) = range_pair
                        operand_range = operand_stop - operand_start
                        instr_range = instr_stop - instr_start
                        if operand_range == 1 and instr_range == 1:
                            # 1-1 correspondence
                            instr_idx = instr_start
                            operand_idx = operand_start
                            instr_ax = instr_axes[instr_idx]
                            operand_ax = operand_axes[operand_idx]
                        elif operand_range == 1:
                            # N-1 correspondence, pick the best match
                            # from the multiple instr axes to assign
                            for i in range(instr_start, instr_stop):
                                instr_ax = instr_axes[instr_start]
                                instr_idx = i
                                if instr_ax is not None:
                                    break
                            operand_idx = operand_start
                            operand_ax = operand_axes[operand_idx]
                        elif instr_range == 1:
                            # N-1 correspondence, pick the best match
                            # from the multiple operand axes to assign
                            for i in range(operand_start, operand_stop):
                                operand_ax = operand_axes[operand_start]
                                operand_idx = i
                                if operand_ax is not None:
                                    break
                            instr_idx = instr_start
                            instr_ax = instr_axes[instr_idx]
                        else:
                            # N:M corresponding, can't match these axes
                            instr_ax = None
                            operand_ax = None

                        # we cannot assign a single logical sharding
                        # to multiple tensor indices, hence the range == 1
                        if (
                            operand_ax is None
                            and instr_ax is not None
                            and operand_range == 1
                        ):
                            if isinstance(operand_axes, tuple):
                                operand_axes = list(operand_axes)
                            operand_axes[operand_idx] = instr_ax

                        # we cannot assign a single logical sharding
                        # to multiple tensor indices, hence the range == 1
                        if (
                            instr_ax is None
                            and operand_ax is not None
                            and instr_range == 1
                        ):
                            if isinstance(instr_axes, tuple):
                                instr_axes = list(instr_axes)
                            instr_axes[instr_idx] = operand_ax

            elif instr.opcode in _ELEMENTWISE_OPS:
                for operand_id in instr.operand_ids:
                    operand = all_instructions[operand_id]
                    operand_axes = known_axes[operand_id]
                    for idx, (instr_ax, operand_ax) in enumerate(
                        zip(instr_axes, operand_axes)
                    ):
                        if instr_ax is None and operand_ax is not None:
                            instr_axes[idx] = operand_ax
                            updated = True
                        if operand_ax is None and instr_ax is not None:
                            if isinstance(operand_axes, tuple):
                                operand_axes = list(operand_axes)
                                known_axes[operand_id] = operand_axes
                            operand_axes[idx] = instr_ax
                            updated = True
                    verify_size(operand, operand_axes, instr)

            elif instr.opcode == "while":
                instr_axes = _recurse_to_comp(
                    instr,
                    instr.called_computation_ids[0],
                    known_axes[instr.operand_ids[0]],
                )

            elif instr.opcode == "dynamic-slice":
                operand_axes = known_axes[instr.operand_ids[0]]
                for i, (op_ax, instr_ax) in enumerate(
                    zip(operand_axes, instr_axes)
                ):
                    if instr_ax is None and op_ax is not None:
                        instr_axes[i] = op_ax
                    if operand_ax is None and instr_ax is not None:
                        operand_axes[i] = instr_ax

            elif instr.opcode == "conditional":
                _recurse_to_comp(
                    instr,
                    instr.called_computation_ids[0],
                    known_axes[instr.operand_ids[1]],
                )
                instr_axes = _recurse_to_comp(
                    instr,
                    instr.called_computation_ids[1],
                    known_axes[instr.operand_ids[2]],
                )

            elif instr.opcode == "call":
                called_comp = all_computations[instr.called_computation_ids[0]]
                params = [
                    called_instr
                    for called_instr in called_comp.instructions
                    if called_instr.opcode == "parameter"
                ]
                comp_known_axes = {}
                for idx, param in enumerate(params):
                    operand_id = instr.operand_ids[param.parameter_number]
                    comp_known_axes[param.id] = known_axes[operand_id]
                    operand = all_instructions[operand_id]
                comp_known_axes = _sharding_propagation_helper(
                    called_comp,
                    all_computations,
                    comp_known_axes,
                    is_entry_computation=False,
                )
                comp_axes = comp_known_axes[called_comp.root_id]
                _fill_nones(instr.shape, instr_axes, comp_axes)

            verify_size(instr, instr_axes, instr)
            known_axes[instr.id] = instr_axes

    return known_axes


def compute_sharding_propagation(
    comp: hlo_pb2.HloComputationProto,
    all_computations: Mapping[int, hlo_pb2.HloComputationProto],
):
    known_axes = {}
    return _sharding_propagation_helper(comp, all_computations, known_axes)
