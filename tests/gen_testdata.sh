#! /usr/bin/env bash

txtfile=dump_gen/module_0000.jit_step_fn.before_optimizations.txt
pbfile=dump_gen/module_0000.jit_step_fn.before_optimizations.hlo.pb

examples="constant_output dot_layers implicit_decomposition many_layers microbatches pruned simple_layers"

rm -rf dump_gen
for example in $examples
do
  legate examples/jax/$example.py --xla_dump_to=dump_gen
  legate examples/jax/$example.py --xla_dump_to=dump_gen --xla_dump_hlo_as_proto

  cp dump_gen/module_0000.*.before_optimizations.hlo.pb testdata/$example.pb
  cp dump_gen/module_0000.*.before_optimizations.txt testdata/$example.txt

  rm -rf dump_gen
done

gen_attention_data() {

legate examples/jax/$1.py \
    --nbatch 4 \
    --batch 32 \
    --seq 256 \
    --hidden 512 \
    --nlayers 4 \
    --nreps 8 \
    --dump-to=dump_attention \
    --dump-hlo-txt \
    --abstract

cp dump_attention/module_0000.*.before_optimizations.hlo.pb testdata/$1.pb
cp dump_attention/module_0000.*.before_optimizations.txt testdata/$1.txt
rm -rf dump_attention

}

gen_attention_data attention
gen_attention_data attention-mp
gen_attention_data attention-while-loop
gen_attention_data attention-while-loop-dp
gen_attention_data attention-ckpt

