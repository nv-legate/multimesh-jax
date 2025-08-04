#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def in_jupyter():
    try:
        from IPython import get_ipython

        return "IPKernelApp" in get_ipython().config
    finally:
        return False


if in_jupyter():
    from IPython.display import HTML, display

    display(HTML("<style>.container { width:100% !important; }</style>"))
else:
    matplotlib.use("agg")


def get_cuda_df(con, start: int, end: int):
    query = f"""
    SELECT
      start, 
      end,
      end - start AS duration,
      streamId,
      deviceId,
      S.value AS short_name, 
      T.value AS full_name
    FROM CUPTI_ACTIVITY_KIND_KERNEL AS C
    JOIN StringIds AS S
      ON C.shortName == S.id
    JOIN StringIds AS T
      ON C.demangledName == T.id
    WHERE deviceId == 0
     AND start > {start}
     AND end < {end}
    ORDER BY start
    """  # noqa:
    cuda_df = pd.read_sql_query(query, con)
    repl = {
        "full_name": "text",
    }

    cols = [repl.get(x, x) for x in cuda_df.columns]
    cuda_df.columns = cols
    return cuda_df


def get_zuku_domain_id(con):
    # First, figure out the domain ID for all the Zuku events
    query = """
    SELECT *
    FROM NVTX_EVENTS AS N
    WHERE text LIKE "%zuku%"
    """
    zdf = pd.read_sql_query(query, con)
    zuku_domain_id = zdf.iloc[0].domainId
    return zuku_domain_id


def get_zuku_df(con, start: int, end: int, domain_id=None):
    if domain_id is None:
        domain_id = get_zuku_domain_id(con)

    query = f"""
    SELECT
        N.*,
        end - start AS duration,
        S.value AS region_name
    FROM NVTX_EVENTS AS N
    FULL JOIN StringIds AS S
      ON S.id == N.textId
    WHERE domainId == {domain_id}
        AND start > {start}
        AND END < {end}
    """  # noqa
    nvtx_df = pd.read_sql_query(query, con)
    return nvtx_df


def get_step_markers(con, domain_id=None):
    if domain_id is None:
        domain_id = get_zuku_domain_id(con)

    query = f"""
    SELECT
        N.*,
        S.value AS region_name
    FROM NVTX_EVENTS AS N
    FULL JOIN StringIds AS S
      ON S.id == N.textId
    WHERE domainId == {domain_id}
        AND END IS NULL
        AND text LIKE "%jit_autoshard_train%"
    """  # noqa
    df = pd.read_sql_query(query, con)
    return df


def get_start_time(con, domain_id=None):
    if domain_id is None:
        domain_id = get_zuku_domain_id(con)

    query = f"""
    SELECT
        N.*,
        S.value AS region_name
    FROM NVTX_EVENTS AS N
    FULL JOIN StringIds AS S
      ON S.id == N.textId
    WHERE domainId == {domain_id}
        AND END IS NULL
        AND text LIKE "%start zuku%"
    """  # noqa

    df = pd.read_sql_query(query, con)
    try:
        return df.iloc[0].start
    finally:
        # if no start marker is found, just use zero
        return 0


def get_boundaries(con, step_markers=None, domain_id=None):
    if step_markers is None:
        step_markers = get_step_markers(con, domain_id)

    t_after_warmup = (
        step_markers[step_markers["text"].str.contains("start")]
        .start.nsmallest(2)
        .iloc[1]
    )
    t_done = step_markers["start"].max()
    return t_after_warmup, t_done


def label_steps(con, df, step_markers=None, boundaries=None, domain_id=None):
    if domain_id is None:
        domain_id = get_zuku_domain_id(con)

    if step_markers is None:
        step_markers = get_step_markers(con, domain_id)

    step_markers = step_markers[
        step_markers.text.str.contains("start")
    ].reset_index()

    if boundaries is None:
        boundaries = get_boundaries(con, step_markers, domain_id)

    max_idx = len(step_markers) - 1

    def find_step(row):
        for idx, srow in step_markers.iterrows():
            if srow.start > row.start:
                return idx - 1
        return max_idx

    df["step"] = df.apply(find_step, axis=1)
    df["microbatch"] = df.groupby(["step", "text"]).cumcount()
    df["microbatch"] = df.apply(
        lambda x: x.microbatch if "loop" in x.text else -1, axis=1
    )
    return df


def get_summary_df(cuda_df, zuku_df):
    """
    Given input DataFrames `cuda_df` and `zuku_df` containing
    CUDA events and zuku NVTX events, creates a summary dataframe
    showing the total time attributed to:
    * Compute
    * Exposed NCCL communication
    * Idle time
    * Reshard time between task
    * Idle time between tasks
    For each column above, there is a row corresponding to:
    * Current observed times
    * Target times with optimization
    * SoL times with all idle/exposed comm removed
    """
    events = []
    START = 0
    END = 1

    CUDA = 0
    ZUKU = 1
    for index, row in cuda_df.iterrows():
        events.append((row.start, START, row.text, row.streamId, CUDA))
        events.append((row.end, END, row.text, row.streamId, CUDA))

    for index, row in zuku_df.iterrows():
        events.append((row.start, START, row.text, None, ZUKU))
        events.append((row.end, END, row.text, None, ZUKU))

    events.sort()

    last_timestamp = None

    num_compute_active = 0
    num_nccl_ag_active = 0
    num_nccl_ar_rs_active = 0
    num_nccl_other_active = 0
    num_tasks_active = 0
    num_reshard_active = 0

    total = 0
    compute_time = 0
    nccl_ag_time = 0
    nccl_ar_rs_time = 0
    nccl_other_time = 0
    cuda_idle_time = 0
    reshard_time = 0
    pipeline_idle_time = 0

    for timestamp, typ, name, stream, kind in events[:]:
        if last_timestamp:
            duration = timestamp - last_timestamp
            total += duration
            # attribute the time to whatever
            # activity is currently
            # if ANY compute kernel is active
            # then treat it as compute time
            # even if NCCL or other kernels are running
            if num_compute_active > 0:
                compute_time += duration
            elif num_nccl_ag_active > 0:
                nccl_ag_time += duration
            elif num_nccl_ar_rs_active > 0:
                nccl_ar_rs_time += duration
            elif num_nccl_other_active > 0:
                nccl_other_time += duration
            elif num_tasks_active > 0:
                cuda_idle_time += duration
            elif num_reshard_active > 0:
                reshard_time += duration
            else:
                pipeline_idle_time += duration

        if typ == START:
            inc = 1
        else:
            inc = -1

        if kind == CUDA:
            if "nccl" in name:
                if "Gather" in name:
                    num_nccl_ag_active += inc
                elif "Redu" in name:
                    num_nccl_ar_rs_active += inc
                else:
                    num_nccl_other_active += inc
            else:
                num_compute_active += inc
        else:
            if "reshard" in name:
                num_reshard_active += inc
            else:
                num_tasks_active += inc

        last_timestamp = timestamp

    compute_ms = compute_time * 1e-6
    nccl_ag_ms = nccl_ag_time * 1e-6
    nccl_ar_ms = nccl_ar_rs_time * 1e-6
    nccl_other_ms = nccl_other_time * 1e-6
    cuda_idle_ms = cuda_idle_time * 1e-6
    reshard_ms = reshard_time * 1e-6
    pipeline_idle_ms = pipeline_idle_time * 1e-6
    total_ms = total * 1e-6

    print(f"Total         = {total_ms:.2f}ms")  # noqa: E221
    print(f"Compute       = {compute_ms:.2f}ms")  # noqa: E221
    print(f"NCCL AG       = {nccl_ag_ms:.2f}ms")  # noqa: E221
    print(f"NCCL AR/RS    = {nccl_ar_ms:.2f}ms")  # noqa: E221
    print(f"NCCL Other    = {nccl_other_ms:.2f}ms")  # noqa: E221
    print(f"CUDA Idle     = {cuda_idle_ms:.2f}ms")  # noqa: E221
    print(f"Reshard       = {reshard_ms:.2f}ms")  # noqa: E221
    print(f"Pipeline Idle = {pipeline_idle_ms:.2f}ms")  # noqa: E221

    totals = {
        "Compute": np.array([compute_ms, compute_ms]),
        "NCCL AG": np.array([nccl_ag_ms, 0]),
        "NCCL AR/RS": np.array([nccl_ar_ms, 0]),
        "NCCL Other": np.array([nccl_other_ms, 0]),
        "CUDA idle": np.array([cuda_idle_ms, 0]),
        "Reshard": np.array([reshard_ms, 0]),
        "PP idle": np.array([pipeline_idle_ms, 0]),
    }

    return pd.DataFrame(index=["Current", "SoL"], data=totals)


db = sys.argv[1]

con = sqlite3.connect(db)

# figure out the Zuku domain ID from the NVTX_EVENTS table
query = """
SELECT *
FROM NVTX_EVENTS AS N
WHERE text LIKE "%zuku%"
"""
zdf = pd.read_sql_query(query, con)
zuku_domain_id = zdf.iloc[0].domainId

# determine the start and end of the training steps
# ignoring a warmup phase on the first step
start, end = get_boundaries(con, domain_id=zuku_domain_id)

cuda_df = get_cuda_df(con, start, end)
zuku_df = get_zuku_df(con, start, end, domain_id=zuku_domain_id)

s_df = get_summary_df(cuda_df, zuku_df)

# ensure that each zuku task intersects at most two bins
# by making the bin size the max task length
bin_size = zuku_df["duration"].max()
zuku_df["bin"] = zuku_df["start"] // bin_size
zuku_df["idx"] = zuku_df.index
num_bins = zuku_df["bin"].max() + 1


# create a searchable index so that we can assign every CUDA
# event to its enclosing zuku task
h = zuku_df.copy()
h = h.set_index(["bin", "idx"])


# not every bin number will exist in the index
# so we need to create a get() function that defaults to None
def get_bin(h, b):
    try:
        return h.loc[b]
    except KeyError:
        return None


bins = [get_bin(h, b) for b in range(num_bins)]


# search all zuku tasks within the bin
# zuku tasks and cuda events can touch at most 2 bins
# based on the definition of the bin size
def get_matching_zuku_region(row):
    bin = row.start // bin_size
    if bin < num_bins:
        for idx, zrow in bins[bin].iterrows():
            if zrow.start <= row.start and zrow.end >= row.end:
                return idx
    # the owning zuku task might be in the prev bin
    for idx, zrow in bins[bin - 1].iterrows():
        if zrow.start <= row.start and zrow.end >= row.end:
            return idx
    raise Exception("no matching zuku task found for CUDA kernel")


cuda_df["idx"] = cuda_df.apply(get_matching_zuku_region, axis=1)

# normalize all times to the zuku start time
t_zero_time = get_start_time(con, zuku_domain_id)
zuku_df.start -= t_zero_time
zuku_df.end -= t_zero_time
cuda_df.start -= t_zero_time
cuda_df.end -= t_zero_time

# compute step numbers and microbatch numbers for every task
zuku_df = label_steps(con, zuku_df)
zuku_df = zuku_df[["start", "end", "idx", "text", "step", "microbatch"]]
zuku_df.columns = ["start", "end", "idx", "task", "step", "microbatch"]

cuda_df = cuda_df[["start", "end", "idx", "text"]]
cuda_df.columns = ["start", "end", "idx", "kernel_name"]

# use the matched idx number to join cuda events to their parent zuku task
cuda_df = cuda_df.merge(
    zuku_df[["idx", "task", "step", "microbatch"]], on="idx"
)
cuda_df["kernel_name"] = cuda_df.kernel_name.apply(lambda x: x[:40])

# assign a unique label to equivalent kernels
# equivalent = same type of task, but different microbatch
cuda_df["kernel_id"] = cuda_df.groupby(["step", "microbatch"]).cumcount()

cuda_df.to_parquet("cuda.parquet")
zuku_df.to_parquet("zuku.parquet")
s_df.to_csv("summary.csv")

cuda_df.info()

# plot the summary bar chart
width = 0.5
fig, ax = plt.subplots()
bottom = np.zeros(len(s_df))

for i, col in enumerate(s_df.columns):
    p = ax.bar(s_df.index, s_df[col], width, label=col, bottom=bottom)
    bottom += np.array(s_df[col])

ax.set_ylabel("Time (ms)")

y_offset = 100
# Add labels to each bar.
for i, row_name in enumerate(s_df.index):
    total_ms = s_df.loc[row_name].sum()
    ax.text(
        row_name,
        total_ms + y_offset,
        f"{total_ms:.0f} ms",  # noqa: E231
        ha="center",
        weight="bold",
    )

ax.legend(bbox_to_anchor=(0.9, 0.5))
plt.savefig("summary.png")

if in_jupyter():
    plt.show()
