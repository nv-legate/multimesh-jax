#! /usr/bin/env bash

for example in \
 "attention 2 async" 
do
  set -- $example
  echo "Running $1"
  echo "$2 GPUS"
  legate examples/hlo.py \
    --$3 \
    --fbmem 14000 \
    --gpus $2 \
    --gdb \
    --load-only \
    -level legate.llm=1 \
    --hlo testdata/$1.pb \
    --gin testdata/$1.gin \
    || exit 1
done

echo "All tests passed"



