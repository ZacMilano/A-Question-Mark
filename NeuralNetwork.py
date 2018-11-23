import AbeTF as tf
import numpy as np
from Data import *
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

  Making this numerically stable by adding a normalization factor (the max
  value in the list).
  '''
  if not derivative:
    mod_list = np.exp(inputs - np.max(inputs))
    return mod_list/float(sum(mod_list))
  else:
    # i-th row in jacobian is gradient of i-th softmax dimension
    # i.e. derivative of i-th softmax dimension wrt each input variable.
    # j-th col in i-th row is partial derivative of i-th softmax dimension wrt
    # the j-th input variable.
    s = softmax(inputs, derivative=False)
    jacobian = []
    for i in range(len(inputs)):
      jacobian.append([])
      for j in range(len(inputs)):
        jacobian[i].append(-1 * s[i] * s[j] if i == j
                           else (s[i] * (1 - s[j])))
    return np.array(jacobian)

def cross_entropy(outputs, labels, epsilon=1e-9, derivative=False):
  '''
  Compute the cross-entropy error on the output vs the labels (desired output).

  outputs is a (batch_size, output_size)-shaped np.array, where each row has
  been pushed through softmax() (at the end).

  labels is a sublist of labels for the inputs that caused outputs to be what
  it is. It is a python list of shape (batch_size, 1).
  '''
  # l exists just because outputs is a 2D array, and we want to index through
  # all the examples while choosing a subset for the second dimension
  l = len(labels)
  if not derivative:
    lst_log_likelihood = -1*np.log(np.abs(outputs[range(l), labels] + epsilon))
    total_loss = np.sum(lst_log_likelihood)/l
    return total_loss
  else:
    gradient = outputs.copy()
    gradient[range(l), labels] -= 1 # Subtract one from those that count
    return gradient/l # Normalize

def expected(n):
  '''
  Return list of length 62, where the n-th value is 1 and rest is 0.
  n is the expected class label of an image.

  We probably won't use this.
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

    # self.train_labels = [expected(label) for label in self.train.labels()]
    self.train_labels = self.train.labels()
    self.train_images = self.train.images()

    # self.test_labels = [expected(label) for label in self.test.labels()]
    self.test_labels = self.test.labels()
    self.test_images = self.test.images()

    # Hyperparameters
    self.input_size = 28*28
    self.hidden_size = 20
    self.output_size = 62
    self.num_hidden_layers = 2
    self.learning_rate = 0.001
    self.batch_size = 500
    self.epochs = 1

    self.hidden_activation = sigmoid
    self.final_activation = softmax

    self._create_weights_and_biases()

    print('Neural network created.')

  def __call__(self):
    pass

  def _create_weights_and_biases(self):
    '''
    Create weight matrices and bias vectors for layer-to-layer transition.
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

  def train(self):
    '''
    Perform all the training batches, and make a graph or something.
    '''
    for iteration in range(np.ceil(len(self.train_labels) / self.batch_size)):
      self.train_batch(iteration)
      if iteration % 100 in [0, np.ceil(len(self.train_labels) / self.batch_size)]:
        # Add a point on a graph or something. This is after 50000 (100*500)
        # training examples! (or perhaps is just the last iteration)
        pass

  def train_batch(self, batch_number):
    '''
    Perform a forward-propagation pass, training some stuff
    '''
    first_ind = self.batch_size * batch_number
    last_ind = self.batch_size * (1 + batch_number)
    for i in range(first_ind, last_ind):
      if i >= len(self.train_labels):
        break
      # Do da training boiiii
    print('haha i am training, trust me comrade')

  def backpropagation(self, outputs, labels):
    '''
    Implement gradient descent algorithm to minimize error.
    '''
    # https://en.wikipedia.org/wiki/Backpropagation#Example_loss_function
    # Use cross-entropy as error measure
    E = cross_entropy(outputs, labels)
    dE_da3 = lambda i: i
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
