import tensorflow as tf
zero_out_module = tf.load_op_library('./layer_op.so')

inm = tf.constant([[1,2,3,5],[1,3,4,5],[1,4,5,7]], dtype=tf.float32)
par = tf.constant([1,0,0,1], dtype=tf.float32)

print inm.dtype

with tf.Session(''):
  ret = zero_out_module.layer_op(
    inm, par
  ).eval()

print ret.dtype
print ret
