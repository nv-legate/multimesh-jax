#! /usr/bin/env bash

for example in \
 "attention 2 async" 
do
  set -- $example
  echo "Running $1"
  echo "$2 GPUS"
  legate \
    --fbmem 14000 \
    --gpus $2 \
    examples/hlo.py \
    -level legate.llm=1 \
    --distributed \
    --hlo testdata/$1.pb \
    --gin testdata/$1.gin \
    --$3 \
    || exit 1
done

echo "All tests passed"



