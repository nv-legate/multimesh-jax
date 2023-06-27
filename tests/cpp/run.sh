#!/bin/bash

cd build
ctest --gpus 2 --verbose -rP  --output-on-failure
