# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

# Specifying setup.py makes a plugin installable via a Python package manager.
# `entry_points` is an important field makes plugins discoverable by TensorBoard
# at runtime.
# See https://packaging.python.org/specifications/entry-points/
setuptools.setup(
    name="tensors",
    version="0.1.0",
    description="Sample TensorBoard plugin.",
    packages=["tensors"],
    package_data={"tensors": ["static/**"],},
    entry_points={
        "tensorboard_plugins": [
            "tensors = tensors.plugin:TensorsPlugin",
        ],
    },
)
