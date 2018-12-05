import AbeTF as Atf
import tensorflow as tf
from Data import *
from time import time
from helpers import normalize_v2, expected
import numpy as np

def x_entropy(actual, predicted):
  # x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
  x_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels = actual,
    logits = predicted,
    name   = 'x_entropy'
  )
  return tf.reduce_mean(x_ent, name='x_entropy_loss')

class CNN(Atf.Model):
  name = 'cnn'

  IMAGE_WIDTH = IMAGE_HEIGHT = 28
  N_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

  INPUT_SHAPE = [N_PIXELS]

  def __init__(self, x_dim=(28*28), y_dim=62, training=True,
               loss_factory=x_entropy):
    super().__init__(x_dim=x_dim, y_dim=y_dim, loss_factory=loss_factory,
                     training=training)

  def new_conv_layer(self, inputs=None, kernels=None, biases=None):
    # We want only named args so that there is less confusion.
    strides = [1, 1, 1, 1]
    padding = 'VALID'

    convolved_images = tf.nn.conv2d(inputs, kernels, strides=strides,
                                    padding=padding)
    convolved_images = tf.nn.relu(convolved_images + biases)
    return convolved_images

  def pool(self, convolved, pooling_shape=(2,2), stride=1):
    ksize = [1, *pooling_shape, 1] # pooling_shape must be 2-tuple of ints >0
    strides = [1, stride, stride, 1]
    padding = 'SAME'

    convolved = tf.nn.max_pool(convolved, ksize=ksize, strides=strides,
                               padding=padding)
    return convolved

  def define_variables(self):
    '''Define vars needed for model.'''
    with tf.variable_scope('conv_0'):
      filter_shape = (5,5)
      n_filters = 40
      k_shape = [filter_shape[0], filter_shape[1], 1, n_filters]

      self.k_0 = tf.get_variable(
        'kernels', shape=k_shape, initializer=tf.random_normal_initializer())
      self.b_0  = tf.get_variable(
        'biases', shape=[n_filters], initializer=tf.zeros_initializer())
      # Due to VALID padding
      resulting_image_width = CNN.IMAGE_WIDTH - (filter_shape[0] - 1)

    with tf.variable_scope('conv_1'):
      filter_shape = (5,5)
      n_filters_old = n_filters
      n_filters = 40
      k_shape = [filter_shape[0], filter_shape[1], n_filters_old, n_filters]

      self.k_1 = tf.get_variable(
        'kernels', shape=k_shape, initializer=tf.random_normal_initializer())
      self.b_1  = tf.get_variable(
        'biases', shape=[n_filters], initializer=tf.zeros_initializer())
      # Due to VALID padding
      resulting_image_width = resulting_image_width - (filter_shape[0] - 1)

    with tf.variable_scope('fc_features', reuse=tf.AUTO_REUSE):
      n_features = 150
      W_shape = [n_filters * resulting_image_width * resulting_image_width,
                 n_features]
      self.W_features = tf.get_variable(
        'W', shape=W_shape, initializer=tf.random_normal_initializer())
      self.b_features = tf.get_variable(
        'b', shape=[n_features], initializer=tf.zeros_initializer())

    with tf.variable_scope('fc_classes', reuse=tf.AUTO_REUSE):
      W_shape = [n_features, self.y_dim]
      self.W_final = tf.get_variable(
        'W', shape=W_shape, initializer=tf.random_normal_initializer())
      # FailedPreconditionError (see above for traceback): Attempting to use
      # uninitialized value fc_final/b
      self.b_final = tf.get_variable(
        'b', shape=[self.y_dim], initializer=tf.zeros_initializer())

  def define_model(self):
    '''Defines self.predicted_y based on self.x and any variables.'''
    square_x = tf.reshape(self.x, [-1, CNN.IMAGE_HEIGHT, CNN.IMAGE_WIDTH, 1])
    convolved = self.new_conv_layer(inputs=square_x, kernels=self.k_0,
                                    biases=self.b_0)
    convolved = self.pool(convolved)
    convolved_2 = self.new_conv_layer(inputs=convolved, kernels=self.k_1,
                                      biases=self.b_1)
    convolved_2 = self.pool(convolved_2)

    c_shape = convolved_2.shape
    conv_neurons = tf.reshape(convolved_2,
                              [-1, int(c_shape[1] * c_shape[2] * c_shape[3])])
    features = tf.matmul(conv_neurons, self.W_features) + self.b_features
    self.predicted_y = tf.matmul(features, self.W_final) + self.b_final

  def define_train(self):
    '''Defines self.optimizer and self.training_step based on self.loss.'''
    self.optimizer = tf.train.AdamOptimizer()
    self.training_step = self.optimizer.minimize(self.loss)

if __name__ == '__main__':
  t0 = time()

  try:
    # n = CNN()
    # print('CNN made.')
    d = Data()
    imgs,   labels   = np.array(d.images()[    :4000]), \
        d.labels()[    :4000]
    imgs_t, labels_t = np.array(d.images()[4000:5000]), \
        d.labels()[4000:5000]
    labels, labels_t = np.array([expected(label) for label in labels]), \
        np.array([expected(label) for label in labels_t])
    st = Atf.train_model(CNN, imgs, labels)
    print(Atf.test_model(CNN, st, imgs_t, labels_t))
  except Exception as e:
    raise e

  dt = time() - t0
  mins, secs = int(dt // 60), int(dt % 60)
  print('-'*80 +
        '\nCompleted in {:0>2d}:{:0>2d}\n'.format(mins, secs) +
         '-'*80)
