import AbeTF as tf
import numpy as np
from Data import *
from math import exp
from time import time

def sigmoid(x, derivative=False):
  '''
  Sigmoid activation function.
  Takes in any real number, outputs a number between 0 and 1.
  '''
  if not derivative:
    return 1 / (1  + np.exp(-x))
  else:
    s = sigmoid(x, derivative=False)
    return s * (1 - s)

def softmax(inputs, derivative=False):
  '''
  Softmax activation function.
  Takes in a list of numbers and outputs a new list of numbers with probability
  values summing to 1.
  '''
  return np.exp(inputs)/float(sum(np.exp(inputs)))

def expected(n):
  '''
  Return list of length 62, where the n-th value is 1 and rest is 0.
  n is the expected class label of an image.
  '''
  y = [0] * 62
  y[n] = 1
  return y

def normalize(lst, norm_factor=256):
  '''
  Normalize the given list of numbers (or np.array) into a np.array.
  '''
  return np.array(lst)/norm_factor

class NeuralNetwork:
  def __init__(self, data_directory='../data_samples', final_testing=False):
    if final_testing:
      self.train = Data(data_directory=data_directory, is_test_data=False)
      self.test = Data(data_directory=data_directory, is_test_data=True)
    else:
      self.train, self.test = Data.train_and_pseudo_test()

    print('Data loaded.')

    self.train_labels = [expected(label) for label in self.train.labels()]
    self.train_images = self.train.images()

    self.test_labels = [expected(label) for label in self.test.labels()]
    self.test_images = self.test.images()

    # Hyperparameters
    self.input_size = 28*28
    self.hidden_size = 20
    self.output_size = 62
    self.num_hidden_layers = 2
    self.learning_rate = 0.001
    self.batch_size = 500
    self.epochs = 1

    self._create_weights_and_biases()

    print('Neural network created.')

  def __call__(self):
    pass

  def _create_weights_and_biases(self):
    '''
    Create weight matrices and bias vectors for transition from layer to layer.
    '''
    self.weight_matrices = []
    self.biases = []
    lengths = self.layer_lengths()
    for n in range(len(lengths) - 1):
      self.weight_matrices.append(np.random.rand(lengths[n+1], lengths[n]))
      self.biases.append(np.random.rand(lengths[n+1]))
    print('Weight matrices and bias vectors initialized.')

  def layer_lengths(self):
    '''
    Return the length of each of the layers of the NN in the form of a list.
    [input layer, {...hidden layers...} , output layer]
    '''
    shapes = [self.input_size]
    for i in range(self.num_hidden_layers):
      shapes.append(self.hidden_size)
    shapes.append(self.output_size)
    return shapes

  def train_batch(self, batch_number):
    '''
    Perform a forward-propagation pass, training
    '''
    first_ind = self.batch_size * batch_number
    last_ind = self.batch_size * (1 + batch_number)
    for i in range(first_ind, last_ind):
      if i >= len(self.train_labels):
        break
      # Do da training boiiii
    print('haha i am training, trust me comrade')

  def backpropagation(self):
    '''
    Implement gradient descent algorithm to minimize error.
    '''
    # https://en.wikipedia.org/wiki/Backpropagation#Example_loss_function
    # e = .5*|y - y_bar|^2
    # e = (1/(2n))*sigma(from=0, to=n, ind=i, ||y_i - y_bar_i||^2)
    print('yeet')


if __name__ == '__main__':
  t0 = time()

  n = NeuralNetwork()
  n() # Equivalent to n.__call__()

  dt = time() - t0
  mins, secs = int(dt // 60), int(dt % 60)
  print('-'*80 +
        '\nCompleted in {:0>2d}:{:0>2d}\n'.format(mins, secs) +
        '-'*80)
