#! /usr/bin/env bash

for example in \
 "attention 2 async" \
 "attention-ckpt 2 async" \
 "attention-while-loop 2 async" \
 "constant_output 1 async" \
 "dot_layers 2 async" \
 "many_layers 2 async" \
 "microbatches 2 async" \
 "implicit_decomposition 1 noasync" \
 "pruned 1 noasync" \
 "simple_layers 2 async"
do
  set -- $example
  echo "Running $1"
  echo "$2 GPUS"
  legate \
    --fbmem 14000 \
    --gpus $2 \
    examples/hlo.py \
    -level legate.xla.mapper=1 \
    -level legate.xla=1 \
    --distributed \
    --hlo testdata/$1.pb \
    --gin testdata/$1.gin \
    --$3 \
    || exit 1
done

echo "All tests passed"



