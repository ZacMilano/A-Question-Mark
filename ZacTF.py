import tensorflow as tf

def fully_connected_layer(inputs=None, output_size=None, activation=None,
                          use_bias=True, bias_init=tf.zeros_initializer(),
                          trainable=True, name=None):
  return tf.layers.dense(
    inputs,
    output_size,
    activation       = activation,
    use_bias         = use_bias,
    bias_initializer = bias_init,
    trainable        = trainable,
    name             = name
  )

def convolutional_layer(inputs=None, filters=None, kernel_size=None,
                        padding='SAME', activation=tf.nn.relu, use_bias=True,
                        bias_initializer=tf.zeros_initializer(), trainable=True,
                        name=None):
  return tf.layers.conv2d(
    inputs           = inputs,
    filters          = filters,
    kernel_size      = kernel_size,
    padding          = padding,
    activation       = activation,
    use_bias         = use_bias,
    bias_initializer = tf.zeros_initializer(),
    trainable        = trainable,
    name             = name
  )

def pooling_layer(inputs=None, pool_width=2, stride=1, padding='SAME',
                  name=None):
  return tf.nn.max_pool(
    inputs,
    ksize   = [1, pool_width, pool_width, 1],
    strides = [1, stride,     stride,     1],
    padding = padding,
    name    = name
  )

def init_vars(sess):
  sess.run(tf.global_variables_initializer())

def x_entropy(actual, predicted):
  # x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
  x_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels = actual,
    logits = predicted,
    name   = 'x_entropy'
  )
  return tf.reduce_mean(x_ent, name='x_entropy_loss')

# Tensorflow/CUDA's GPU Session makes the terminal output all crazy :)
def vprint(s):
  print('-'*80 + '\n' + s + '\n' + '-'*80)
