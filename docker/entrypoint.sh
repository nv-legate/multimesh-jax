#! /bin/bash

source /opt/install/miniconda/etc/profile.d/conda.sh
conda activate legere
if [[ -e /opt/mofed-ver ]]; then
    IMAGE_VER="$(cat /opt/mofed-ver)"
    if [[ ! -e /sys/module/mlx5_core/version ]]; then
        echo "Warning: MOFED installation not detected on host machine, multi-node runs may fail" 1>&2
    else
        HOST_VER="$(cat /sys/module/mlx5_core/version)"
        if [[ "$HOST_VER" != "${IMAGE_VER:0:${#HOST_VER}}" ]]; then
            echo "Warning: Host and image MOFED versions differ ($HOST_VER vs $IMAGE_VER), multi-node runs may fail" 1>&2
        fi
    fi
fi
exec "$@"