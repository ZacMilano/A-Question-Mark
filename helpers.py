import numpy as np

def relu(x, derivative=False):
  '''
  ReLU activation function.
  '''
  if not derivative:
    if isinstance(x, np.ndarray):
      return np.array([x_ if x_ >= 0 else 0.2 * x_ for x_ in x])
    else:
      return x if x >= 0 else 0.2 * x
  else:
    if isinstance(x, np.ndarray):
      return np.array([1 if x_ >= 0 else 0.2 for x_ in x])
    else:
      return 1 if x >= 0 else 0.2

def mse(y_hat, target_label, derivative=False):
  '''
  Return mean square error cost.
  '''
  y = expected(target_label)
  if not derivative:
    e = 0

    for y_i in y_hat:
      e += (y_i - y) ** 2
    return e/2
  else:
    # return np.abs(y_hat - y)
    return y_hat - y

def tanh(x, derivative=False):
  '''
  tanh activation function. Squishes R into (-1,1) range.
  '''
  if not derivative:
    return 2 / (1  + np.exp(-2*x)) - 1
  else:
    t = tanh(x, derivative=False)
    return 1 - t * t

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

def cross_entropy(y_hat, target_label, epsilon=1e-9, derivative=False):
  '''
  Compute the cross-entropy error on the output vs the labels (desired output).

  y_hat is a output_size-length np.array, where each row has been pushed through
  softmax() (at the end).

  target_label is an integer that entails the desired target class of the input
  that caused the network to output y_hat.
  '''
  # l exists just because outputs is a 2D array, and we want to index through
  # all the examples while choosing a subset for the second dimension
  if not derivative:
    loss = 0
    for i, y_i in enumerate(y_hat):
      if i == target_label:
        loss += -np.log(y_i)
      else:
        loss += -np.log(1 - y_i)
    return loss
  else:
    grads = [0 if i != target_label
             else -1/(y_hat[i] + epsilon)
             for i in range(len(y_hat))]
    return np.array(grads)

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
