#! /usr/bin/env bash

# these got wrong versions installed
# which leads to import errors when running

conda run -no-capture-out -n legere \
  python -m pip install --force-reinstall --no-deps --upgrade \
  lingvo

conda run --no-capture-out -n legere \
  python -m pip install --force-reinstall --no-deps \
  numpy==1.26.0
