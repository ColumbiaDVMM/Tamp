import tensorflow as tf

from tensorflow.python.framework import ops

struct = tf.load_op_library('../../layer_op.so')

@ops.RegisterGradient("LayerOp")
def _layer_op_grad(op, grad):
  print('grad is called: ', op, grad, op.node_def.name)
  return struct.layer_op_gradient(op.inputs[0], op.inputs[1], op.outputs[0], grad, op_name=op.node_def.name)

@ops.RegisterShape("LayerOp")
def _layer_op_shape(op):
  """Shape function for the ZeroOut op.

  This is the unconstrained version of ZeroOut, which produces an output
  with the same shape as its input.
  """
  return [tf.TensorShape([128, 384])]

@ops.RegisterShape("LayerOpGradient")
def _layer_op_gradient_shape(op):
  return [op.inputs[0].get_shape(), op.inputs[1].get_shape()]
