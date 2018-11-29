import AbeTF as tf
import numpy as np
from Data import *
from time import time
from helpers import sigmoid, softmax, cross_entropy, normalize

class CNN(tf.Model):
  name = 'cnn'

  def __init__(self, training=True):
    super().__init__() # :)
    if training:
      self.train_data, self.test_data = Data.train_and_pseudo_test(proportion=0.9)
    else:
      self.train_data = Data(data_directory=data_directory, is_test_data=False)
      self.test_data = Data(data_directory=data_directory, is_test_data=True)
    print('Data loaded.')

    self.train_labels = self.train_data.labels()
    self.train_images = self.train_data.images()

    self.test_labels = self.test_data.labels()
    self.test_images = self.test_data.images()

  def define_variables(self):
    '''Define vars needed for model.'''
    pass

  def define_model(self):
    '''Defines self.predicted_y based on self.x and any variables.'''
    self.predicted_y = None
    raise NotImplementedError('No implementation for model function.')

  def define_train(self):
    '''Defines self.optimizer and self.training_step based on self.loss.'''
    self.optimizer = None
    self.training_step = None
    raise NotImplementedError('No implementation for training step.')

if __name__ == '__main__':
  t0 = time()

  n = CNN()

  dt = time() - t0
  mins, secs = int(dt // 60), int(dt % 60)
  print('-'*80 +
        '\nCompleted in {:0>2d}:{:0>2d}\n'.format(mins, secs) +
         '-'*80)
