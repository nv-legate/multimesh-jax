#! /usr/bin/env bash

mkdir cache
chmod 0777 cache

docker run -u 1000:1000 -d -v $(pwd)/cache:/data \
	-p 9090:8080 -p 9092:9092 buchgr/bazel-remote-cache \
	--max_size 100
