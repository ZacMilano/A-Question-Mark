import numpy as np

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
