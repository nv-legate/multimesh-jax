#! /usr/bin/env bash

git clone -b control_replication ssh://git@gitlab.com/StanfordLegion/legion.git legion
git clone -b cpp-branch-23.09 ssh://git@github.com/nv-legate/legate.core.internal legate
git clone -b legate-main --filter tree:0 git@github.com:nv-legate/xla.git xla_extension
git clone ssh://git@gitlab-master.nvidia.com:12051/legate/legate-xla-bridge.git legate_xla
git clone git@github.com:nv-legate/jax.git jaxlib
git clone git@github.com:nv-legate/jax.git jax

