# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Script to test TF-TRT INT8 conversion without calibration on Mnist model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tensorrt.python import trt_convert
# pylint: disable=unused-import
from tensorflow.contrib.tensorrt.python.ops import trt_engine_op
# pylint: enable=unused-import
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import data
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
import numpy as np

class DynamicInputShapesTest(test_util.TensorFlowTestCase):

  def _BuildGraph(self, x):
    conv_filter1 = constant_op.constant(np.random.randn(3,3,1,8),
                                        name="weights1",
                                        dtype=dtypes.float32)
    bias1 = constant_op.constant(np.random.randn(8), dtype=dtypes.float32)
    x = nn.conv2d(
        input=x,
        filter=conv_filter1,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name="conv")
    x = nn.bias_add(x, bias1)
    x = nn.relu(x)
    conv_filter2 = constant_op.constant(np.random.randn(3,3,8,1),
                                       name="weights2",
                                       dtype=dtypes.float32)
    bias2 = constant_op.constant(np.random.randn(1), dtype=dtypes.float32)
    x = nn.conv2d(
        input=x,
        filter=conv_filter2,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name="conv")
    x = nn.bias_add(x, bias2)
    x = array_ops.identity(x, name='output')
    return x

  def _GetGraphDef(self, use_trt, max_batch_size):
    """Get the frozen GraphDef.

    Args:
      use_trt: whether use TF-TRT to convert the graph.
      max_batch_size: the max batch size to apply during TF-TRT conversion.

    Returns:
      The frozen GraphDef.
    """
    graph = ops.Graph()
    with self.session(graph=graph) as sess:
      with graph.device('/GPU:0'):
        x = array_ops.placeholder(shape=(None, None, None, 1),
                                  dtype=dtypes.float32,
                                  name='input')
        self._BuildGraph(x)
      # Load weights
      #sess.run(tf.global_variables_initializer())
      # Freeze
      graph_def = graph_util.convert_variables_to_constants(
          sess, sess.graph_def, output_node_names=['output'])
    # Convert with TF-TRT
    if use_trt:
      logging.info('Number of nodes before TF-TRT conversion: %d',
                   len(graph_def.node))
      graph_def = trt_convert.create_inference_graph(
          graph_def,
          outputs=['output'],
          max_batch_size=max_batch_size,
          precision_mode='FP32',
          max_workspace_size_bytes=4096 << 19,
          minimum_segment_size=2,
          is_dynamic_op=True
      )
      logging.info('Number of nodes after TF-TRT conversion: %d',
                   len(graph_def.node))
      num_engines = len(
          [1 for n in graph_def.node if str(n.op) == 'TRTEngineOp'])
      self.assertEqual(1, num_engines)
    return graph_def

  def testDynamicInputShapes(self):
    if not trt_convert.is_tensorrt_enabled():
      return

    graph_def = self._GetGraphDef(use_trt=True, max_batch_size=1)

    graph = ops.Graph()
    with self.session(graph=graph) as sess:
      inp, out = importer.import_graph_def(
          graph_def=graph_def, return_elements=["input", "output"])
      inp = inp.outputs[0]
      out = out.outputs[0]
      input_shapes = [[1, 5, 5, 1],
                      [1, 3, 1, 1],
                      [1, 9, 9, 1],
                      [1, 224, 224, 1],
                      [1, 128, 224, 1]]
      for shape in input_shapes:
        x = np.ones(shape=shape, dtype=np.float32)
        y = sess.run(out, feed_dict={inp: x})
        self.assertAllEqual(shape, y.shape)

if __name__ == '__main__':
  test.main()
