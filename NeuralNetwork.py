from Data import *
from time import time

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
