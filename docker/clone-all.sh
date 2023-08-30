#! /usr/bin/env bash

git clone -b control_replication ssh://git@gitlab.com/StanfordLegion/legion.git legion
git clone -b cpp-branch-23.09 ssh://git@github.com/nv-legate/legate.core.internal legate
git clone --filter tree:0 ssh://git@gitlab-master.nvidia.com:12051/legate/xla.git xla_extension
git clone ssh://git@gitlab-master.nvidia.com:12051/legate/legate-xla-bridge.git legate_xla

