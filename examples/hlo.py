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

import argparse
import os
import re
import sys
import time
from contextlib import contextmanager
from typing import Optional

import gin

import legate.timing as timing
import lllm
from lllm.mesh import legate_global_mesh


@contextmanager
def timer(name) -> float:
    start = time.perf_counter()
    try:
        yield start
    finally:
        lllm.runtime.issue_execution_fence(block=True)
        total = time.perf_counter() - start
        print(f"{name} ran for {total}s")


def run_hlo(
    path: str,
    exec_async: bool,
    load_only: bool,
    dump_output: bool,
    niter: int,
    nwarmup: int,
    coordinator_addr: Optional[str] = None,
    gin_paths: list[str] = [],
    gin_params: list[str] = [],
    print_layers: bool = False,
    alias_inputs_and_outputs: bool = False,
    dry_run: bool = False,
) -> None:
    dry_run = False
    if gin_paths:
        gin.parse_config_files_and_bindings(gin_paths, gin_params)

    device_mesh = legate_global_mesh()
    print("mesh=", device_mesh)
    model = lllm.load_module(path)

    all_tensors = model.create_tensors(match_inputs=alias_inputs_and_outputs)

    unmatched_inputs = {}
    unmatched_outputs = {}
    for param in model.parameters:
        py_id = id(all_tensors.get_parameter(param))
        unmatched_inputs[py_id] = param
    for root in model.roots:
        py_id = id(all_tensors.get_root(root))
        if py_id in unmatched_inputs:
            del unmatched_inputs[py_id]
        else:
            unmatched_outputs[py_id] = root

    for param in unmatched_inputs.values():
        print(f"unmatched input {param.name}: {param.shape.dimensions}")
    for output in unmatched_outputs.values():
        print(f"unmatched output {output.name}: {output.shape.dimensions}")

    if exec_async:
        with timer("async decompose"):
            model.decompose_into_layers(all_tensors)
            model.print_decomposition_tree()
            if print_layers:
                print(model.summary(mesh=device_mesh))

    ts_load = timing.time()

    if coordinator_addr is not None:
        lllm.runtime.init_distributed(coordinator_addr)

    model.load(global_mesh=device_mesh, debug=(load_only or dry_run))

    ts_finish_load = timing.time()
    print(f"module load ran for {(ts_finish_load - ts_load) * 1e-6}s")
    if load_only:
        return

    ts_warmup = timing.time()

    model(
        all_tensors,
        global_mesh=device_mesh,
        init_tensors=True,
        dry_run=dry_run,
    )

    for _ in range(nwarmup):
        model(all_tensors, global_mesh=device_mesh, dry_run=dry_run)

    print(f"Global tensor size is {all_tensors.size/1e9} GB")

    ts_start = timing.time()

    for _ in range(niter):
        model(all_tensors, global_mesh=device_mesh, dry_run=dry_run)

    ts_end = timing.time()

    print(f"warmup ran for {(ts_start - ts_warmup) * 1e-6}s")
    print(f"execute ran for {(ts_end - ts_start) * 1e-6}s")

    if coordinator_addr is not None:
        lllm.runtime.shutdown_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    sys.path.append(os.getcwd())

    parser.add_argument(
        "--hlo",
        type=str,
        required=True,
        dest="path",
        help="Path to HLO module",
    )
    parser.add_argument(
        "--gin",
        type=str,
        dest="gin_paths",
        action="append",
        default=[],
        help="Path to GIN configurations for device mesh",
    )
    parser.add_argument(
        "--gin-param",
        type=str,
        dest="gin_params",
        action="append",
        default=[],
        help="GIN variable specifications",
    )
    parser.add_argument(
        "--coordinator-addr",
        type=str,
        dest="coordinator_addr",
        default=None,
        help="The hostname or address for the distributed coordinator",
    )
    parser.add_argument(
        "--dump-to",
        type=str,
        dest="dump_to",
        default=None,
        help="Path to dump HLO modules",
    )
    parser.add_argument(
        "--dump-hlo-txt",
        action="store_true",
        dest="dump_hlo_txt",
        default=False,
        help="Whether to dump HLO module text",
    )
    parser.add_argument(
        "--dump-hlo-pb",
        action="store_true",
        dest="dump_hlo_pb",
        default=False,
        help="Whether to dump HLO module protobuf",
    )
    parser.add_argument(
        "--dump-hlo-dot",
        action="store_true",
        dest="dump_hlo_dot",
        default=False,
        help="Whether to dump dot output for the HLO module",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        default=False,
        help="Whether to do a dry run that doesnt execute",
    )
    parser.add_argument(
        "--async",
        action="store_true",
        default=False,
        dest="exec_async",
        help="Turn on asynchronous execution using Legate",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        dest="distributed",
        help="Initialized distributed coordinator",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        default=False,
        dest="load_only",
        help="Test only the model construction (for debugging)",
    )
    parser.add_argument(
        "--dump-output",
        action="store_true",
        default=False,
        dest="dump_output",
        help="Print outputs",
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        dest="niter",
        default=1,
        help="Number of iterations",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        dest="nwarmup",
        default=0,
        help="Number of extra warm-up iterations",
    )
    parser.add_argument(
        "-p",
        "--print-layers",
        action="store_true",
        default=False,
        dest="print_layers",
        help="Print details of HLO Module layer decomposition",
    )
    parser.add_argument(
        "--alias-inputs",
        action="store_true",
        default=False,
        dest="alias_inputs",
        help="Alias inputs to the best matching output",
    )
    args, _ = parser.parse_known_args()

    xla_flags = []
    if args.dump_to is not None:
        xla_flags.append(f"--xla_dump_to={args.dump_to}")

    if args.dump_hlo_txt:
        xla_flags.append("--xla_dump_hlo_as_text")

    if args.dump_hlo_pb:
        xla_flags.append("--xla_dump_hlo_as_proto")

    if args.dump_hlo_dot:
        xla_flags.append("--xla_dump_hlo_as_dot")

    if xla_flags:
        os.environ["XLA_FLAGS"] = " ".join(xla_flags)

    coordinator_addr = args.coordinator_addr
    if args.distributed and coordinator_addr is None:
        slurm_nodelist = os.environ.get("SLURM_STEP_NODELIST")
        if slurm_nodelist is not None:
            first_comma = slurm_nodelist.find(",")
            first_bracket = slurm_nodelist.find("[")

            if first_bracket == -1 and first_comma == -1:
                # single node abc102
                coordinator_addr = slurm_nodelist
            elif first_bracket == -1:
                # comma-separated list abc102,xyz103
                coordinator_addr = slurm_nodelist.split(",")[0]
            elif first_comma == -1 or first_comma > first_bracket:
                # A few possible forms or permutations thereof
                # 1) abc-[102,103]
                # 2) abc-[102-103],xyz
                root, first_brackets = slurm_nodelist.split("[")[:2]
                # inside brackets is 102-103 or 102,103
                # split on , and - and take the first ID
                first_id = re.split(r",|-", first_brackets)[0]
                coordinator_addr = root + first_id
            else:
                # Comma is the first delimiter
                coordinator_addr = slurm_nodelist.split(",")[0]
        else:
            coordinator_addr = "localhost"
    sys.stderr.write(f"Coordinator={coordinator_addr}\n")
    sys.stderr.flush()

    run_hlo(
        args.path,
        args.exec_async,
        args.load_only,
        args.dump_output,
        args.niter,
        args.nwarmup,
        coordinator_addr,
        args.gin_paths,
        args.gin_params,
        args.print_layers,
        args.alias_inputs,
        args.dry_run,
    )
