#! /usr/bin/env bash

./tags.sh
docker build --network=host --add-host host.docker.internal:172.17.0.1 -t legate-jax -f Dockerfile.multi-stage --target paxml_image .

