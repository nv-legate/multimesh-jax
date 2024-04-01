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

checkout legion     control_replication ssh://git@gitlab.com/StanfordLegion/legion.git 9d0c95fe2a080
checkout legate     cpp-branch-24.01    ssh://git@github.com/nv-legate/legate.core.internal 97037d54864
checkout xla        legate-main         ssh://git@github.com/nv-legate/xla.git        590d67f2fc1
checkout legate-jax main                ssh://git@github.com/nv-legate/legate.jax.git 755946ed25
checkout jaxlib     legate-main         ssh://git@github.com/nv-legate/jax.git        dfe51d3f105
checkout jax        legate-main         ssh://git@github.com/nv-legate/jax.git        dfe51d3f105
checkout paxml      main                https://github.com/google/paxml.git           cc904d3   paxml.patch
checkout praxis     main                https://github.com/google/praxis.git          545e00a   praxis.patch
checkout orbax      main                https://github.com/google/orbax.git           4d372c1
checkout clu        main                https://github.com/google/CommonLoopUtils.git f30bc44
checkout optax      main                https://github.com/google-deepmind/optax.git  a49564e
checkout flax       main                https://github.com/google/flax.git            d58e6dde
checkout seqio      main                https://github.com/google/seqio.git           513d1fe
checkout chex       master              https://github.com/google-deepmind/chex.git   a3f1dd0

