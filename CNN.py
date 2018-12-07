from time import time

import numpy as np
import tensorflow as tf
import cv2

from Data import Data
from helpers import normalize_tf
from ZacTF import vprint
import ZacTF as ztf

# def main_definitions():
vars_to_store = []
WIDTH = HEIGHT = 28
N_PX = WIDTH * HEIGHT
K = 62 # Number of classes
LR = 0.0005 # Learning rate

error = ztf.x_entropy

X = tf.placeholder(tf.float32, shape=[None, N_PX])
x_n = normalize_tf(X)
x_as_image = tf.reshape(x_n, [-1, HEIGHT, WIDTH, 1])

# 1ST CONVOLUTIONAL LAYER + POOLING {{{
F = 40
S = 5
conv_0 = ztf.convolutional_layer(inputs=x_as_image,
                                 filters=F,
                                 kernel_size=(S,S),
                                 name='conv_0')

pool_width = 2
stride     = 2

pool_0 = ztf.pooling_layer(inputs=conv_0,
                           pool_width=pool_width,
                           stride=stride,
                           name='max_pool_0')
# }}}

# 2ND CONVOLUTIONAL LAYER + POOLING {{{
F = 50
S = 5
conv_1 = ztf.convolutional_layer(inputs=pool_0,
                                 filters=F,
                                 kernel_size=(S,S),
                                 name='conv_1')

pool_width = 2
stride     = 2

pool_1 = ztf.pooling_layer(inputs=conv_1,
                           pool_width=pool_width,
                           stride=stride,
                           name='max_pool_1')
# }}}

s = pool_1.shape
flat_len = s[1] * s[2] * s[3]
# Flatten the convolved features
convolved_flattened = tf.reshape(pool_1, [-1, flat_len])


# Make feature neuron layer
D = 1050 # Number of 'feature' neurons
features = ztf.fully_connected_layer(inputs=convolved_flattened, output_size=D,
                                     activation=tf.nn.relu, name='W_b_features')

# Output
Y_hat = ztf.fully_connected_layer(inputs=features, output_size=K, name='y_hat')
predict_class = tf.argmax(Y_hat, 1)
# Desired output
desired_class = tf.placeholder(tf.uint8, shape=(None,), name='desired_class')
Y = tf.one_hot(desired_class, K)

# Value to minimize
loss = error(Y, Y_hat)
correct_classification = tf.equal(tf.cast(desired_class, tf.int64), predict_class)
accuracy = tf.reduce_mean(tf.cast(correct_classification, tf.float32))

optimizer  = tf.train.AdamOptimizer(learning_rate=LR, epsilon=0.001)
train_step = optimizer.minimize(loss)

def batch(b, batch_size, imgs, labels):
  start = batch_size * b
  end   = batch_size * (b+1)
  batch_imgs, batch_labels = imgs[start:end], labels[start:end]
  return batch_imgs, batch_labels

def evaluate_model(session=None, test_imgs=None, test_labels=None):
  batch_size     = 200 # Makes the computation spatially stable on my computer
  n_test_batches = int(np.ceil(len(test_labels)/batch_size))
  accs, losses   = [], []

  for b in range(n_test_batches):
    batch_test_imgs, batch_test_labels = batch(b, batch_size, test_imgs,
                                               test_labels)
    feed_dict = {
      X             : batch_test_imgs,
      desired_class : batch_test_labels
    }
    result = session.run([accuracy, loss], feed_dict=feed_dict)
    accs.append(  result[0])
    losses.append(result[1])
  acc   = sum(accs)  /len(accs)
  loss_ = sum(losses)/len(losses)
  print('Test accuracy: {0:.2f}%\nLoss:          {1}'.format(acc*100, loss_))
  return acc, loss_

def make_prediction(session=None, test_img=None):
  y_ = session.run(predict_class, feed_dict = {X : [test_img]})
  return int(y_[0])

def train(session=None, batch_size=None, imgs=None, labels=None,
          test_imgs=None, test_labels=None):
  B = int(np.ceil(len(labels)/batch_size))
  for b in range(B):
    if b%100 == 0: evaluate_model(session=session, test_imgs=test_imgs,
                                  test_labels=test_labels)
    batch_imgs, batch_labels = batch(b, batch_size, imgs, labels)
    feed_dict = {
      X             : batch_imgs,
      desired_class : batch_labels
    }
    session.run(train_step, feed_dict=feed_dict)

def store_var(v):
  # Perhaps instead later find vars with tf.GraphKeys.TRAINABLE_VARIABLES?
  vars_to_store.append(v)

def flatten(img):
  # flatten a square image.
  res = []
  for row in img:
    res += row
  return res

def main(training):
  # main_definitions()
  if training:
    d, d_t = Data.train_and_pseudo_test()

  else:
    # In this case you're actually training but with more data lol
    d, d_t = Data(), Data(is_test_data=True)

  vprint('Data loaded.')

  imgs,      labels      = d.images(),   d.labels()
  test_imgs, test_labels = d_t.images(), d_t.labels()

  vprint('Opening tf session...')
  with tf.Session() as sess: # Context manager automatically closes it!
    vprint('Initializing global variables...')
    ztf.init_vars(sess) # just runs tf.global_variables_initializer()
    # TRAIN THAT BAD BOI
    train(session=sess, batch_size=500, imgs=imgs, labels=labels,
          test_imgs=test_imgs, test_labels=test_labels)
    test = input('Want to test it with an image?  ')
    yes = ('y', 'yes')
    no = ('n', 'no')
    while test.lower() not in no:
      img = cv2.imread("../character.png", 0)
      img = img.tolist()
      img = flatten(img)
      y_ = make_prediction(session=sess, test_img=img)
      print('Hmm... I think that character is {}.'.format(d.label_display(y_)))
      test = input('Want to try again?  ')

if __name__ == '__main__':
  yes = ('y', 'yes')
  training = input('Are you training your model?  ').lower() in yes
  main(training)
