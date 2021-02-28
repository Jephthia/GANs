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
    ], dtype=tf.float32) * -1.
    
    with writer.as_default():
        tf.summary.write("mymatrix3", mat, step=0, metadata=tf.compat.v1.SummaryMetadata(plugin_data=tf.compat.v1.SummaryMetadata.PluginData(content=b"", plugin_name="tensors")))


if __name__ == "__main__":
    app.run(main)
