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
    return 1 / (1  + exp(-x))
  else:
    s = sigmoid(x, derivative=False)
    return s * (1 - s)

def expected(n):
  '''
  Return list of length 62, where the n-th value is 1 and rest is 0.
  '''
  y = [0] * 62
  y[n] = 1
  return y

class NeuralNetwork:
  def __init__(self, data_directory='../data_samples', final_testing=False,
               num_hidden=2, hidden_length=20, learning_rate=0.001):
    if final_testing:
      self.train = Data(data_directory=data_directory, is_test_data=False)
      self.test = Data(data_directory=data_directory, is_test_data=True)
    else:
      self.train, self.test = Data.train_and_pseudo_test()

    self.train_labels = np.array(self.train.labels())
    self.train_images = np.array(self.train.images())

    self.test_labels = np.array(self.test.labels())
    self.test_images = np.array(self.test.images())

    # Hyperparameters
    self.num_hidden_layers = num_hidden
    self.hidden_layer_length = hidden_length
    self.learning_rate = learning_rate


if __name__ == "__main__":
  time_initial = time()
  n = NeuralNetwork()
  dt = time() - time_initial
  mins, secs = int(dt // 60), int(dt % 60)
  print("-"*80 +
        "\nCompleted in {:0>2d}:{:0>2d}\n".format(mins, secs) +
        "-"*80)
