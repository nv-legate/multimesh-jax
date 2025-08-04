#! /usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

export LC_ALL=C.UTF-8

python -m pip install -r docs/requirements.txt
sphinx-build -a -b html -D nb_execution_mode=off docs docs/_build/html -j auto
