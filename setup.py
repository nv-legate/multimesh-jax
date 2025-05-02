#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_namespace_packages
from skbuild import setup

import versioneer

setup(
    name="MultiMesh JAX",
    version=versioneer.get_version(),
    description="MultiMesh for Jax MPMD execution plugin",
    url="TBD",
    author="NVIDIA Corporation",
    license="Closed source",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_namespace_packages(include=["jax_plugins.*", "multimesh.*"]),
    include_package_data=True,
    package_data={
        "multimesh.jax": ["*.so"],
    },
    cmdclass=versioneer.get_cmdclass(),
    install_requires=["numpy>=1.22", "pybind11[global]", "gin"],
    zip_safe=False,
)
