#! /usr/bin/env python

import os

debug = 5
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # LOG(INFO)
os.environ["TF_CPP_MAX_LOG_LEVEL"] = str(debug)
os.environ["TF_CPP_VMODULE"] = f"hlo_partition={debug}"

import sys

import legate.jax

hlo_path = sys.argv[1]
gin_config = sys.argv[2]

legate.jax.init(config=gin_config)


legate.jax.compile_hlo_module(
    hlo_path, num_partitions=32, erase_sharding=True, platform="gpu"
)
