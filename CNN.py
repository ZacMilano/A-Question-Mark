import AbeTF as Atf
import tensorflow as tf
from Data import *
from time import time
from helpers import normalize_v2

def x_entropy(actual, predicted):
  x_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels = actual,
    logits = predicted,
    name   = 'x_entropy'
  )
  loss = tf.reduce_mean(x_ent, name='x_entropy_loss')
  return loss

class CNN(Atf.Model):
  name = 'cnn'

  IMAGE_WIDTH = IMAGE_HEIGHT = 28
  N_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

  INPUT_SHAPE = [N_PIXELS]
  OUTPUT_SHAPE = [62]

  def __init__(self, training=True, loss_factory=x_entropy):
    super().__init__(x_dim=(28*28), y_dim=62)
    if training:
      self.train_data, self.test_data = \
          Data.train_and_pseudo_test(proportion=0.9)
    else:
      self.train_data = Data(data_directory=data_directory, is_test_data=False)
      self.test_data = Data(data_directory=data_directory, is_test_data=True)
    print('Data loaded.')

    self.train_labels = self.train_data.labels()
    self.train_images = self.train_data.images()

    self.test_labels = self.test_data.labels()
    self.test_images = self.test_data.images()

  def new_conv_layer(self, inputs=None, kernels=None, biases=None):
    strides = [1, 1, 1, 1]
    padding = "SAME"

    convolved_images = tf.nn.conv2d(inputs, kernels, strides=strides,
                                    padding=padding)
    convolved_images = tf.nn.relu(convolved_images + biases)
    return convolved_images

  def pool(self, convolved, pooling_shape=(2,2), stride=2):
    ksize = [1, pooling_shape[0], pooling_shape[1], 1]
    strides = [1, stride, stride, 1]
    padding = 'SAME'

    convolved = tf.nn.max_pool(convolved, ksize=ksize, strides=strides,
                               padding=padding)
    return convolved

  def define_variables(self):
    '''Define vars needed for model.'''
    with tf.variable_scope('conv_0'):
      filter_shape = (3,3)
      input_channels = CNN.N_PIXELS
      n_filters = 40
      shape = [filter_shape[0], filter_shape[1], input_channels, n_filters]

      self.k_0 = tf.get_variable(
        'kernels', shape=shape, initializer=tf.random_normal_initializer())
      self.b_0  = tf.get_variable(
        'biases', shape=[n_filters], initializer=tf.zeros_initializer())

    with tf.variable_scope('fc_final', reuse=tf.AUTO_REUSE):
      shape = [n_filters] + CNN.OUTPUT_SHAPE
      self.W_final = tf.get_variable(
        'W', shape=shape, initializer=tf.random_normal_initializer())
      self.b_final = tf.get_variable(
        'b', shape=CNN.OUTPUT_SHAPE, initializer=tf.zeros_initializer())

  def define_model(self):
    '''Defines self.predicted_y based on self.x and any variables.'''
    convolved = self.new_conv_layer(inputs=self.x, kernels=self.k_0,
                                    biases=self.b_0)
    self.predicted_y = convolved @ self.W_final + self.b_final

  def define_train(self):
    '''Defines self.optimizer and self.training_step based on self.loss.'''
    self.optimizer = tf.train.AdamOptimizer()
    self.training_step = self.optimizer.minimize(self.loss)

if __name__ == '__main__':
  t0 = time()

  n = CNN()

  dt = time() - t0
  mins, secs = int(dt // 60), int(dt % 60)
  print('-'*80 +
        '\nCompleted in {:0>2d}:{:0>2d}\n'.format(mins, secs) +
         '-'*80)
