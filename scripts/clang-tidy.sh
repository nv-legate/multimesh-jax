#! /usr/bin/env bash

pushd /opt/workspace/xla

nproc=${1:-16}

find xla/pjrt/legate -name "*.cc" \
   ! -name "*test*.cc" \
   ! -name "*cu.cc" \
   ! -name "*xla_compiler.cc" \
   -print0 | xargs -0 -P ${nproc} -I % sh -c 'echo % ; clang-tidy-17 -p compile_commands.json %'

if [ $? -ne 0 ]; then
    echo "At least one clang-tidy command failed"
    popd
    exit 1
fi

popd


