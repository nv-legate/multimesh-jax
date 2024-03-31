#! /usr/bin/env bash

echo "Tags" > tags
for d in */ ; do
  pushd $d
  if [ -d .git ]; then
    tag=`git rev-parse HEAD`
    echo "$d $tag" >> ../tags
  fi
  popd
done
