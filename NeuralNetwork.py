import AbeTF as tf
from Data import *
from math import exp
from time import time

def sigmoid(x, derivative=False):
  if not derivative:
    return 1 / (1  + exp(-x))
  else:
    return sigmoid(x, derivative=False) * (1 - sigmoid(x, derivative=False))

class NeuralNetwork:
  def __init__(self):
    self.train, self.test = Data.train_and_pseudo_test()

if __name__ == "__main__":
  time_initial = time()
  n = NeuralNetwork()
  dt = time() - time_initial
  mins, secs = int(dt // 60), int(dt % 60)
  print("-"*80 +
        "\nCompleted in {:0>2d}:{:0>2d}\n".format(mins, secs) +
        "-"*80)
