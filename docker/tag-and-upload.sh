#! /usr/bin/env bash

#tag=`date +"%Y%m%d"`
name=nvcr.io/nvidian/legion/legate-jax-ucx-12.2.2-ubunutu22.04-dev:latest

docker tag legate-jax-dev:latest $name
docker push $name
