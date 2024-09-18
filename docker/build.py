#! /usr/bin/env python

import argparse
import os
import subprocess as sp

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument(
    "--image",
    type=str,
    choices=["opt", "dev"],
    help="Whether to build a development image with all repos or a size-optimized release image",  # noqa: E501
    default="dev",
)

parser.add_argument(
    "--cache", action=argparse.BooleanOptionalAction, default=True
)

parser.add_argument(
    "--cache-port",
    type=str,
    default="172.17.0.1",
    help="the port to use for the Bazel remote cache",
)

parser.add_argument(
    "--cuda-version",
    type=str,
    default="12.4.1",
    help="The CUDA version to use for the base CUDA image",
)

parser.add_argument(
    "--cudnn-version",
    type=str,
    default="",
    help="The cuDNN version to use for the base CUDA image",
)

parser.add_argument(
    "--custom-nccl", action=argparse.BooleanOptionalAction, default=False
)

parser.add_argument(
    "--build-type",
    type=str,
    choices=["Release", "Debug", "RelWithDebInfo"],
    default="Release",
    help="The type of CMake build to execute for Legate repos",
)

parser.add_argument(
    "--tag",
    type=str,
    default="latest",
    help="The tag to use for naming the image",
)
parser.add_argument(
    "--latest", action=argparse.BooleanOptionalAction, default=False
)

parser.add_argument(
    "--repo",
    type=str,
    default="gitlab-master.nvidia.com:5005/legate/quickstart.internal",
    help="The repo (prefix) to use for image uploads",
)

parser.add_argument(
    "--upload", action=argparse.BooleanOptionalAction, default=False
)

args = parser.parse_args()

os.system("./tags.sh")

cmds = ["docker", "build"]

if args.cache:
    cmds.append("--network=host")
    cmds.append("--add-host")
    cmds.append(f"host.docker.internal:{args.cache_port}")

if args.image == "dev":
    dockerfile = "Dockerfile"
    short_image_name = "legate-jax-dev"
else:
    dockerfile = "Dockerfile.multi-stage"
    short_image_name = "legate-jax"

cmds.append("-t")
cmds.append(short_image_name)

cmds.append("-f")
cmds.append(dockerfile)
cmds.append(".")

for arg, value in (
    ("LEGATE_BUILD_TYPE", args.build_type),
    ("CUDA_VERSION", args.cuda_version),
    ("CUDNN_VERSION", args.cudnn_version),
    ("CUSUTOM_NCCL", 1 if args.custom_nccl else 0),
):
    cmds.append("--build-arg")
    cmds.append(f"{arg}={value}")

print(" ".join(cmds))

output = sp.check_output(cmds)
print(output)

full_image_name = f"{short_image_name}:{args.tag}"
os.system(f"docker tag {short_image_name} {full_image_name}")

if args.upload:
    image = f"{args.repo}/{full_image_name}"
    os.system(f"docker tag {short_image_name} {image}")
    os.system(f"docker push {image}")
    if args.latest:
        latest_image = f"{args.repo}/{short_image_name}:latest"
        os.system(f"docker tag {short_image_name} {latest_image}")
        os.system(f"docker push {latest_image}")
