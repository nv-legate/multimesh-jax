#! /usr/bin/env bash

txtfile=dump_gen/module_0000.jit_step_fn.before_optimizations.txt
pbfile=dump_gen/module_0000.jit_step_fn.before_optimizations.hlo.pb

#examples="constant_output dot_layers implicit_decomposition many_layers microbatches pruned replicate simple_layers"
examples=dot_layers

rm -rf dump_gen
for example in $examples
do
  legate examples/jax/$example.py --xla_dump_to=dump_gen
  legate examples/jax/$example.py --xla_dump_to=dump_gen --xla_dump_hlo_as_proto

  cp dump_gen/module_0000.*.before_optimizations.hlo.pb testdata/$example.pb
  cp dump_gen/module_0000.*.before_optimizations.txt testdata/$example.txt

  rm -rf dump_gen
done
#
#legate examples/jax/attention.py \
#    --nbatch 1 \
#    --batch 32 \
#    --seq 256 \
#    --hidden 512 \
#    --nlayers 4 \
#    --nreps 8 \
#    --dump-to=dump_attention \
#    --dump-hlo-txt \
#    --abstract
#
#cp dump_attention/module_0000.*.before_optimizations.hlo.pb testdata/attention.pb
#cp dump_attention/module_0000.*.before_optimizations.txt testdata/attention.txt
#rm -rf dump_attention

legate examples/jax/attention-while-loop.py \
    --nbatch 4 \
    --batch 32 \
    --seq 256 \
    --hidden 512 \
    --nlayers 4 \
    --nreps 8 \
    --dump-to=dump_attention \
    --dump-hlo-txt \
    --abstract

cp dump_attention/module_0000.*.before_optimizations.hlo.pb testdata/attention-while.pb
cp dump_attention/module_0000.*.before_optimizations.txt testdata/attention-while.txt
rm -rf dump_attention
