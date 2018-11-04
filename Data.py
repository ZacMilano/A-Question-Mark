from mnist import MNIST

class Data:
  '''
  Wrapper class for containing EMNIST data.
  '''
  def __init__(self, data_directory="../data_samples", is_test_data=False):
    '''
    Constructor for a Data instance.

    Parameters:
      self : The instance itself
      data_directory : Optional argument indicating the relative path to the
        compressed data, in .gz format; default is a directory above the current
        working directory with name "data_samples"
      is_test_data : Optional argument indicating whether or not this instance
        of data should be used for testing; default is false (so the data will
        be used instead for training). The dataset is already split into
        training and test data, so we just have to call the appropriate data
        retrieval method later.
    '''
    # MNIST/EMNIST data organizer
    emnist_data = MNIST(data_directory)
    # Use EMNIST data (instead of normal MNIST data)
    emnist_data.select_emnist('byclass')

    # Default is to retrieve training data (dataset already has separated
    # training data from test data)
    if not is_test_data:
      self._images, self._labels = emnist_data.load_training()
    else:
      self._images, self._labels = emnist_data.load_testing()

  def images(self):
    '''Return the unlabeled image data'''
    return self._images

  def labels(self):
    '''Return the labels of the data'''
    return self._labels

if __name__ == "__main__":
  print("\nWhy are you running this file..?\n")
