from mnist import MNIST

class Data:
  '''
  Wrapper class for containing EMNIST data.
  '''
  # TODO: Process official_test_data, i.e. subdivide training data
  def __init__(self, data_directory="../data_samples", is_test_data=False,
               official_test_data=False):
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
      official_test_data : Optional argument indicating whether or not this
        instance of data should be used for publication of results; default is
        false because we do not want to overfit to our already-separated test
        data. We will thus instead subdivide our training data.

    Download the data from this link, clicking on "The database in original
    MNIST format":
      https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist

    Extract the .zip file to wherever, and move all emnist-byclass-* files to
    the relative directory data_directory.
    '''
    self.data_directory = data_directory

    # MNIST/EMNIST data organizer
    emnist_data = MNIST(self.data_directory)
    # Use EMNIST data (instead of normal MNIST data)
    emnist_data.select_emnist('byclass')

    # Default is to retrieve training data (dataset already has separated
    # training data from test data)
    if not is_test_data:
      self._images, self._labels = emnist_data.load_training()
    else:
      self._images, self._labels = emnist_data.load_testing()

  def images(self):
    '''
    Return the unlabeled image data.
    Return type is List<List<Integer>>

    Each List<Integer> inside the main list is a list of length 28*28=784,
    where each integer in that list is in the range [0,256), representing an
    activation value of a pixel in a 28*28px image of a character.
    '''
    return self._images

  def labels(self):
    '''Return the labels of the data.
    Return type is List<Integer> where each integer is in the range [0,62).'''
    return self._labels

  def label_display(self, class_label):
    '''
    Return the actual character that was written.
    Return type: String (or None, if no matching class is found)

    Parameter:
      self : The instance of Data itself
      class_label : Integer in range [0,62)

    Data.labels() returns a list of integers, in the range [0,62). [0,9] is
    digits 0-9, [10,35] is A, B, ... , Z, and [36,61] is a, b, ... , z.

    This method maps the class label to a string of length 1 that contains just
    the desired label. Included with the dataset was a .txt file with a proper
    mapping.

    It's not necessary to have this as an instance method, but that's ok.
    '''
    with open(self.data_directory +
              '/emnist-byclass-mapping.txt') as mapping_file:
      # Each line is of this format:
      # "{class #} {ASCII code mapping to character's name}"
      for line in mapping_file:
        label, cls = line.split(" ")
        # Convert strings to integers
        label = int(label)
        cls = int(cls)
        # Return ascii code of mapping
        if class_label == label:
          return chr(cls)

if __name__ == "__main__":
  print("\nWhy are you running this file..?\n")
