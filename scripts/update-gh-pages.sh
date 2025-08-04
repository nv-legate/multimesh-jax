#! /usr/bin/env bash

version=$1

pushd docs

# clear any existing docs
rm -rf _build/html

# link the to-be generated html folder
# to the gh-pages branch as a worktree
mkdir -p _build
pushd _build
git worktree add -f html gh-pages
popd

popd

./scripts/gen-docs.sh

pushd docs/_build/html

git add -u
git commit -m "update to version ${1}"
git push origin gh-pages

popd
