#! /usr/bin/env bash

cuda_version=12.4.1
cudnn_version=
build_type=Release
label=`echo $build_type | tr '[:upper:]' '[:lower:]'`

./tags.sh
docker build \
 --network=host --add-host host.docker.internal:172.17.0.1 \
 -t legate-jax-dev-${label} \
 -f Dockerfile \
 --build-arg LEGATE_BUILD_TYPE=${build_type} \
 --build-arg CUDA_VERSION=${cuda_version} \
 --build-arg CUDNN_VERSION=${cudnn_version} \
 .


name=gitlab-master.nvidia.com:5005/legate/quickstart.internal/legate-jax-ucx-${cuda_version}-ubunutu22.04-dev-${label}:latest
docker tag legate-jax-dev-${label}:latest $name
docker push $name
