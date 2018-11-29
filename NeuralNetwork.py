# Standard libraries
from time import time
# 3rd party libraries
import numpy as np
# Our files
from Data import *
from helpers import mse, normalize, normalize_v2, sigmoid, relu, tanh

class NeuralNetwork:
  def __init__(self, data_directory='../data_samples', final_testing=False,
               load_from_directory=None):
    if final_testing:
      self.train_ = Data(data_directory=data_directory, is_test_data=False)
      self.test_ = Data(data_directory=data_directory, is_test_data=True)
    else:
      self.train_, self.test_ = Data.train_and_pseudo_test(proportion=0.9)

    print('Data loaded.')

    self.train_labels = self.train_.labels()
    self.train_images = self.train_.images()

    self.test_labels = self.test_.labels()
    self.test_images = self.test_.images()

    # Hyperparameters
    self.input_size = 28*28
    self.hidden_size = 20
    self.output_size = 62
    self.num_hidden_layers = 2
    self.learning_rate = .5
    self.batch_size = 250
    self.epochs = 3
    # Want this to be called instead of referenced. This is nice when using the
    # Python shell and altering hyperparameters of the network (so we don't have
    # to update both self.batches AND self.batch_size when we update
    # self.batch_size)
    self.batches = lambda: int(np.ceil(len(self.train_labels)/self.batch_size))

    self.activation = tanh
    self.cost = mse

    if load_from_directory is None:
      self._create_weights_and_biases()
    else:
      self._load_model()

    print('Neural network created.')

  def _create_weights_and_biases(self):
    '''
    Create weight matrices and bias vectors for layer-to-layer transition.
    '''
    self.weight_matrices = []
    self.bias_vectors = []
    lengths = self._layer_lengths()
    for n in range(len(lengths) - 1):
      # Random values between -1 and 1
      self.weight_matrices.append(-1+2*np.random.rand(lengths[n+1], lengths[n]))
      self.bias_vectors.append(-1+2*np.random.rand(lengths[n+1]))
    print('Weight matrices and bias vectors initialized.')

  def _load_model(self, load_from_directory):
    '''
    Load weight matrices and bias vectors from files in directory
    load_from_directory.
    '''
    # Use np.genfromtxt('filename.csv', delimiter=',')
    raise NotImplementedError

  def save_model(self, save_to='../EMNIST_NN_model'):
    '''
    Save model's current state to files in save_to.

    Save to directory outside of git repo.
    '''
    # Use np.savetxt('filename.csv', each_np_array, delimiter=',')
    raise NotImplementedError

  def _layer_lengths(self):
    '''
    Return the length of each of the layers of the NN in the form of a list.
    [input layer, {...hidden layers...} , output layer]
    '''
    shapes = [self.input_size]
    for _ in range(self.num_hidden_layers):
      shapes.append(self.hidden_size)
    shapes.append(self.output_size)
    return shapes

  def feed_forward(self, x):
    '''
    Give output of neural network with current weights and biases if input is x.
    len(x) = self.input_size.

    Output is np.array with size self.output_size, and has been pushed through
    self.activation.
    '''
    last = len(self.weight_matrices) - 1
    y = normalize_v2(np.copy(x))
    for W, b in zip(self.weight_matrices, self.bias_vectors):
      z = np.dot(W, y) + b
      y = self.activation(z)
    return y

  def train(self):
    '''
    Perform all the training batches, and make a graph or something.
    '''
    print('Beginning training...')
    for e in range(self.epochs):
      print('Training epoch #{:2d}...'.format(e))
      performance = [] # Add self.test() values here every once in a while

      for n_batch in range(self.batches()):
        self._train_batch(n_batch, e)
        if n_batch % 500 == 0:
          t = self.test()
          print('After training {0:5d} batches, performance is {1:.2f}%.'\
                .format(n_batch + e*self.batches(), t*100))
          performance.append(t)
    t = self.test()
    print('Training complete. Final accuracy on test data is {0:.2f}%.'.format(
      t*100))
    performance.append(t)

  def _train_batch(self, batch_number, epoch):
    '''
    Perform a forward-propagation pass, training some stuff
    '''
    n = self.batch_size
    k = self.learning_rate / (1 + batch_number/self.batches()) + 0.25*epoch)
    first_ind = n * batch_number
    last_ind = n * (batch_number + 1)
    dE_dW = [np.zeros(W.shape) for W in self.weight_matrices]
    dE_dB = [np.zeros(b.shape) for b in self.bias_vectors]
    for i in range(first_ind, last_ind):
      if i >= len(self.train_labels):
        # Don't want an exception to be raised when training the last batch!
        break
      else: # In the clear
        input_image  = normalize_v2(self.train_images[i])
        target_label = self.train_labels[i]
        _dW, _dB = self.backpropagation(input_image, target_label)
        dE_dW = [dW + dW_ for dW, dW_ in zip(dE_dW, _dW)]
        dE_dB = [dB + dB_ for dB, dB_ in zip(dE_dB, _dB)]

    # k is learning rate, n is for averaging
    self.weight_matrices = [W - (k * dW / n) for W, dW in
                            zip(self.weight_matrices, dE_dW)]
    self.bias_vectors = [b - (k * dB / n) for b, dB in
                         zip(self.bias_vectors, dE_dB)]

  def backpropagation(self, input_image, target_label):
    '''
    Compute gradients for weight matrices and bias vectors based on one
    training example.

    Return gradient of error function with respect to each weight matrix and
    bias vector, as a 2-tuple where the first value is a list of grad values
    for each weight matrix and the second value is a list of grad values for
    each bias vector.
    '''
    n = 1 + self.num_hidden_layers + 1 # input + num_hidden + output
    dW = [np.zeros(W.shape) for W in self.weight_matrices]
    dB = [np.zeros(b.shape) for b in self.bias_vectors]
    a = input_image
    a_vals = [a] # Store neuron activation values
    z_vals = [] # Store neuron values before passing thru activation function
    # This does nearly the same thing as self.feed_forward, but it stores the
    # a- and z-values in lists
    for W, b in zip(self.weight_matrices, self.bias_vectors):
      z_i = np.dot(W, a) + b
      z_vals.append(z_i)
      a = self.activation(z_i)
      a_vals.append(a)

    # dE_da = cross_entropy(a_vals[-1], target_label, derivative=True)
    dE_da = self.cost(a_vals[-1], target_label, derivative=True)
    da_dz = self.activation(z_vals[-1], derivative=True)
    dE_dz = dE_da * da_dz
    dW[-1] = np.dot(np.transpose(np.array([dE_dz])), np.array([a_vals[-2]]))
    dB[-1] = dE_dz # * 1

    # Recursively & iteratively compute dE/dW and dE/dB
    for i in range(2, n):
      next_ = -i + 1  # Layer that curr_ influences
      prev_ = -i - 1  # Layer that influences curr_
      curr_ = -i      # Current layer

      z_i = z_vals[curr_]
      dE_dz = np.dot(np.transpose(self.weight_matrices[next_]), dE_dz) * \
          self.activation(z_i, derivative=True)
      # dE/dW
      dW[curr_] = np.dot(np.transpose(np.array([dE_dz])),
                         np.array([a_vals[prev_]]))
      # dE/dB
      dB[curr_] = dE_dz # * 1

    # print(dW[1], '\n', dB[1])
    return (dW, dB)

  def test(self, test_on_all=False):
    '''
    Test how well the network currently performs on the test data.

    Return a proportion of correct answers to test sample size.
    '''
    if not test_on_all:
      l = self.test_labels
      i = self.test_images
    else:
      l = self.test_labels + self.train_labels
      i = self.test_images + self.train_images
    print('Testing...')
    correct = 0
    for label, image in zip(l, i):
      if np.argmax(self.feed_forward(image)) == label:
        correct += 1
    return correct/len(l)


if __name__ == '__main__':
  t0 = time()

  n = NeuralNetwork()
  n.train()

  to_save = input('Would you like to save this model to directory' +
                  '../EMNIST_NN_model/ ?')
  y = ('y', 'yes', 'yea boi')
  if to_save.lower()[:7] in y: # Allow for longer 'yea boiiiii's
    n.save_model()
  else:
    print('Not saving model.')

  dt = time() - t0
  mins, secs = int(dt // 60), int(dt % 60)
  print('-'*80 +
        '\nCompleted in {:0>2d}:{:0>2d}\n'.format(mins, secs) +
        '-'*80)
