#!/usr/bin/env python3

# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from setuptools import find_namespace_packages
from skbuild import setup

import versioneer

setup(
    name="Legate JAX",
    version=versioneer.get_version(),
    description="Legate-Jax MPMD execution plugin",
    url="TBD",
    author="NVIDIA Corporation",
    license="Closed source",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    packages=find_namespace_packages(include=["jax_plugins.*", "legate.*"]),
    include_package_data=True,
    package_data={
        "legate.jax": ["*.so"],
    },
    cmdclass=versioneer.get_cmdclass(),
    install_requires=["numpy>=1.22", "pybind11[global]", "gin"],
    zip_safe=False,
)
