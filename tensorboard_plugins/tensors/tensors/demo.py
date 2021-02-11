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
"""Demo code.

This generates summary logs viewable by the raw scalars example plugin.
After installing the plugin (`python setup.py develop`), you can run TensorBoard
with logdir set to the `demo_logs` directory.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
import tensorflow as tf


tf.compat.v1.enable_eager_execution()
tf = tf.compat.v2


def main(unused_argv):
    writer = tf.summary.create_file_writer("logs")

    mat = tf.constant([
        [ [ [1111, 1112, 1113, 1114], [1121, 1122, 1123, 1124], [1131, 1132, 1133, 1134] ], [ [1211, 1212, 1213, 1214], [1221, 1222, 1223, 1224], [1231, 1232, 1233, 1234] ] ],
        [ [ [2111, 2112, 2113, 2114], [2121, 2122, 2123, 2124], [2131, 2132, 2133, 2134] ], [ [2211, 2212, 2213, 2214], [2221, 2222, 2223, 2224], [2231, 2232, 2233, 2234] ] ]
    ], dtype=tf.float32)
    
    with writer.as_default():
        tf.summary.write("mymatrix1", mat, step=0)


if __name__ == "__main__":
    app.run(main)
