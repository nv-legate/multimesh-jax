#! /usr/bin/env bash

build_type=Release
label=`echo $build_type | tr '[:upper:]' '[:lower:]'`

./tags.sh
docker build \
 --network=host --add-host host.docker.internal:172.17.0.1 \
 -t legate-jax-dev-${label} \
 -f Dockerfile \
 --build-arg LEGATE_BUILD_TYPE=${build_type} \
 .


name=gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-jax-ucx-12.2.2-ubunutu22.04-dev-${label}:latest
docker tag legate-jax-dev-${label}:latest $name
docker push $name
