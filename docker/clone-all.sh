#! /usr/bin/env bash

function checkout {
  folder=$1
  branch=$2
  repo=$3
  commit=$4
  patch=$5
  if [[ -d "$folder" ]]; then
    pushd $folder
    echo "fetching updates for $folder"
    git fetch origin
    if [[ -z "$commit" ]]; then
      echo "updating $branch of $repo to top-of-tree for $folder"
      git checkout origin/$branch
    else
      echo "checking out commit $commit in $folder"
      git checkout origin/$branch
    fi
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
  fi
  popd
}

checkout chex       master              https://github.com/google-deepmind/chex.git   a3f1dd0
checkout clu        main                https://github.com/google/CommonLoopUtils.git f30bc44
checkout flax       main                https://github.com/google/flax.git            5694517 flax.patch
checkout jax        legate-main         ssh://git@github.com/nv-legate/jax.git        bc414e394
checkout jaxlib     legate-main         ssh://git@github.com/nv-legate/jax.git        bc414e394
checkout zuku       main                ssh://git@gitlab-master.nvidia.com:12051/jwilke/realm-poc.git
checkout legate-jax realm-refactor      ssh://git@github.com/nv-legate/legate.jax.git d9744be6a6
checkout realm      master              https://gitlab.com/StanfordLegion/legion.git  6da07108
checkout optax      main                https://github.com/google-deepmind/optax.git  a49564e
checkout orbax      main                https://github.com/google/orbax.git           1e06498bfb
checkout paxml      main                https://github.com/google/paxml.git           bd0590f843   paxml.patch
checkout praxis     main                https://github.com/google/praxis.git          81154d8e       praxis.patch
checkout seqio      main                https://github.com/google/seqio.git           513d1fe
checkout te         main                https://github.com/NVIDIA/TransformerEngine.git  0b303dad4c  te.patch
checkout xla        realm-refactor      ssh://git@github.com/nv-legate/xla.git        8cbf0cc21
checkout nccl       master              https://github.com/NVIDIA/nccl.git            6203b4c95d71
checkout maxtext    legate-main         git@github.com:mfoerste4/maxtext_priv.git     6b42eab4

pushd te
git submodule update --init --recursive
popd
