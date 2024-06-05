#! /usr/bin/env bash

function checkout {
  folder=$1
  branch=$2
  repo=$3
  commit=$4
  patch=$5
  if [[ -d "$folder" && -z "$commit" ]]; then
    echo "updating $branch of $repo to top-of-tree for $folder"
    pushd $folder
    git fetch origin
    git checkout origin/$branch
    popd
  else
    echo "cloning $branch of $repo into $folder"
    git clone -b $branch --filter tree:0 $repo $folder
    pushd $folder
    if [ ! -z "$commit" ]; then
      git checkout $commit
    fi
    if [ ! -z "$patch" ]; then
      git apply ../$patch
    fi
    popd
  fi
}

checkout legion     master              ssh://git@gitlab.com/StanfordLegion/legion.git      2f87f39e91b8c95de816d207ff03fcada152664f
checkout legate     cpp-branch-24.05    ssh://git@github.com/nv-legate/legate.core.internal 5a650f68fbd42c50a7b3d323e8b87f183f733205 legate.patch
checkout xla        legate-main         ssh://git@github.com/nv-legate/xla.git              d7bebc46549cfb8553629f41cd90d2c830e0695f
checkout legate-jax main                ssh://git@github.com/nv-legate/legate.jax.git       102c3342f9c3ba49af908ffc763c2a027d2da47c
checkout jaxlib     legate-main         ssh://git@github.com/nv-legate/jax.git              67afa4c032cdeac7e9d72ba210bd851d2d470bfb
checkout jax        legate-main         ssh://git@github.com/nv-legate/jax.git              67afa4c032cdeac7e9d72ba210bd851d2d470bfb
checkout paxml      main                https://github.com/google/paxml.git                 bd0590f843 paxml.patch
checkout praxis     main                https://github.com/google/praxis.git                c58bcc4e82 praxis.patch
checkout orbax      main                https://github.com/google/orbax.git                 6673d0c
checkout clu        main                https://github.com/google/CommonLoopUtils.git       c50acb7
checkout optax      main                https://github.com/google-deepmind/optax.git        b4acf8e
checkout flax       main                https://github.com/google/flax.git                  fdbc640c   flax.patch
checkout seqio      main                https://github.com/google/seqio.git                 11706e4a1e
checkout chex       master              https://github.com/google-deepmind/chex.git         53ba23c51
checkout te         main                https://github.com/NVIDIA/TransformerEngine.git     1ec33ae11  te.patch

pushd te
git submodule update --init --recursive
popd

