#! /usr/bin/env bash

if [ ! -z "${CUSTOM_NCCL}" ] &&  [ "${CUSTOM_NCCL}" != "0" ]; then
  make -j src.build  NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90a,code=sm_90a"
  make -j pkg.debian.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_90a,code=sm_90a"
  apt remove -y --allow-change-held-packages libnccl2 libnccl-dev
  apt install -y --reinstall ./build/pkg/deb/libnccl2*.deb
  apt -y install --reinstall ./build/pkg/deb/libnccl-dev*.deb
fi
