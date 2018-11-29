import AbeTF.py as TF
import numpy as np
from Data import *
from time import time
from helpers import sigmoid, softmax, cross_entropy, normalize

class CNN:
  pass

if __name__ == '__main__':
  t0 = time()

  n = CNN()

  dt = time() - t0
  mins, secs = int(dt // 60), int(dt % 60)
  print('-'*80 +
        '\nCompleted in {:0>2d}:{:0>2d}\n'.format(mins, secs) +
        '-'*80)
