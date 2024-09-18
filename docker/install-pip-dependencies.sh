#! /usr/bin/env bash

editable_flag=${1:-}

conda run --no-capture-out -n legere python -m pip install -r /opt/requirements.txt

pushd /opt/chex
conda run -n legere python -m pip install $editable_flag . --no-deps --no-build-isolation --force-reinstall
popd

pushd /opt/flax
conda run -n legere python -m pip install $editable_flag . --no-deps --no-build-isolation --force-reinstall
popd

pushd /opt/clu
conda run -n legere python -m pip install $editable_flag . --no-deps --no-build-isolation --force-reinstall
popd

pushd /opt/optax
conda run -n legere python -m pip install $editable_flag . --no-deps --no-build-isolation --force-reinstall
popd

pushd /opt/seqio
conda run -n legere python -m pip install $editable_flag . --no-deps --no-build-isolation --force-reinstall
popd

pushd /opt/orbax/checkpoint
conda run -n legere python -m pip install $editable_flag . --no-deps --no-build-isolation --force-reinstall
popd

# praxis and paxml must always be installed editable since the
# contrib subfolders are not properly configured for installation

pushd /opt/praxis
conda run -n legere python -m pip install -e . --no-deps --no-build-isolation --force-reinstall
popd

pushd /opt/paxml
conda run -n legere python -m pip install -e . --no-deps --no-build-isolation --force-reinstall
popd
