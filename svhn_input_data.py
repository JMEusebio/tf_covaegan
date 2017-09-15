# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading SVHN data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import scipy.io
import os
import pickle
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

#import matplotlib.pyplot as plt
#from pylab import show

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               pernum = 10):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      #if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        #images = images.astype(numpy.float32)
        #images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

    classsize = 10
    indexlist = []
    self._lepochs_completed = 0
    self._lindex_in_epoch = 0
    self._lnum_examples = pernum * classsize

    for i in range(classsize):
      count = 0
      index = 0
      while count < pernum and index < len(images):
        if self._labels[index, i] == 1:
          count += 1
          indexlist.append(index)
        index += 1

    self.limagesns = images[indexlist]
    self.llabelsns = labels[indexlist]
    perm0 = numpy.arange(self._lnum_examples)
    numpy.random.shuffle(perm0)
    self.limages = self.limagesns[perm0]
    self.llabels = self.llabelsns[perm0]

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch

    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]

  def testing_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch

    # Go to the next epoch
    return self._images[0:batch_size], self._labels[0:batch_size]

  def get_labeled(self):  # , pernum, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    '''start = self._index_in_epoch
    classsize = 10
    indexlist = []
    for i in range(classsize):
      count = 0
      index = 0
      while count < pernum and index < len(self.images):
        if self._labels[index, i] == 1:
          count += 1
          indexlist.append(index)
        index +=1
    return self.images[indexlist], self.labels[indexlist]
    '''

    return self.limagesns, self.llabelsns


  def next_lbatch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._lindex_in_epoch
    # Shuffle for the first epoch
    if start + batch_size > self._lnum_examples:
      # Get the rest examples in this epoch
      rest_num_examples = self._lnum_examples - start
      images_rest_part = self.limages[start:self._lnum_examples]
      labels_rest_part = self.llabels[start:self._lnum_examples]
      start = 0
      self._lindex_in_epoch = batch_size - rest_num_examples
      end = self._lindex_in_epoch
      images_new_part = self.limages[start:end]
      labels_new_part = self.llabels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._lindex_in_epoch += batch_size
      end = self._lindex_in_epoch
      return self.limages[start:end], self.llabels[start:end]

def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   validation_size=5000,
                   color=False,
                   lpernum=10):
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  pickle_file = "./datasets/svhn/pickledsvhn.pickle"
  if not os.path.isfile(pickle_file):
    print ("data not found, pickling datasets")
    train_images, train_labels = load_SVHN("train_32x32.mat")
    extra_images, extra_labels = load_SVHN("extra_32x32.mat")
    test_images, test_labels = load_SVHN("test_32x32.mat")
    train_images = numpy.concatenate((train_images, extra_images), 0)
    train_labels = numpy.concatenate((train_labels, extra_labels), 0)
    del extra_images, extra_labels

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    if (not color):
      train_images = train_images[:, :, :, 0].reshape(
        [train_images.shape[0], train_images.shape[1], train_images.shape[2], 1])
      validation_images = validation_images[:, :, :, 0].reshape(
        [validation_images.shape[0], validation_images.shape[1], validation_images.shape[2], 1])
      test_images = test_images[:, :, :, 0].reshape(
        [test_images.shape[0], test_images.shape[1], test_images.shape[2], 1])

    '''im2show = train_images[100, :, :, :]
    imgplot = plt.imshow(numpy.squeeze(im2show))
    show()'''

    for c in range(train_images.shape[3]):
      for i in range(train_images.shape[0]):
        x = (train_images[i, :, :, c])
        # train_images[i, :, :, 0] = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)) * 2 - 1
        train_images[i, :, :, c] = (x * 2) - 1

      for i in range(validation_images.shape[0]):
        x = (validation_images[i, :, :, c])
        # validation_images[i, :, :, 0] = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)) * 2 - 1
        validation_images[i, :, :, c] = (x * 2) - 1

      for i in range(test_images.shape[0]):
        x = (test_images[i, :, :, c])
        # test_images[i, :, :, 0] = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)) * 2 - 1
        test_images[i, :, :, c] = (x * 2) - 1

    permtest = numpy.arange(test_images.shape[0])
    numpy.random.shuffle(permtest)
    test_images = test_images[permtest]
    test_labels = test_labels[permtest]

    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_images,
        'train_labels': train_labels,
        'valid_dataset': validation_images,
        'valid_labels': validation_labels,
        'test_dataset': test_images,
        'test_labels': test_labels,
      }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to {}: {}'.format(pickle_file, e))
      raise

    print('SVHN Data pickled')
  else:
    with open(pickle_file, 'rb') as f:

      save = pickle.load(f)
      train_images = save['train_dataset']
      train_labels = save['train_labels']
      validation_images = save['valid_dataset']
      validation_labels = save['valid_labels']
      test_images = save['test_dataset']
      test_labels = save['test_labels']
      del save
      print("loaded pickled SVHN")

  '''
  im2show = train_images[500, :, :, :]
  imgplot = plt.imshow(numpy.squeeze(im2show))
  show()
  '''


  '''im2show = train_images[100, :, :, :]
  imgplot = plt.imshow(numpy.squeeze(im2show))
  show()'''



  for i in range(len(train_labels)):
    x = train_labels[i]
    y = numpy.append(x[9], x[:9])
    # print(train_labels[i])
    train_labels[i] = y

  for i in range(len(test_labels)):
    x = test_labels[i]
    y = numpy.append(x[9], x[:9])
    # print(train_labels[i])
    test_labels[i] = y

  for i in range(len(validation_labels)):
    x = validation_labels[i]
    y = numpy.append(x[9], x[:9])
    # print(train_labels[i])
    validation_labels[i] = y

      # print(y)
    # im2show = train_images[i, :, :, :]
    # imgplot = plt.imshow(numpy.squeeze(im2show))
    # show()


    pass

  '''print(train_labels[0])
  im2show = train_images[0, :, :, :]
  imgplot = plt.imshow(numpy.squeeze(im2show))
  show()
  '''


  permval = numpy.arange(validation_images.shape[0])
  numpy.random.shuffle(permval)
  validation_images = validation_images[permval]
  validation_labels = validation_labels[permval]
  validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape, pernum=100)
  del validation_images, validation_labels

  #permtest = numpy.arange(test_images.shape[0])
  #numpy.random.shuffle(permtest)
  #test_images = test_images[permtest]
  #test_labels = test_labels[permtest]
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape , pernum=100)
  del test_images, test_labels

  permtrain = numpy.arange(train_images.shape[0])
  numpy.random.shuffle(permtrain)
  train_images = train_images[permtrain]
  train_labels = train_labels[permtrain]
  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape , pernum=lpernum)
  del train_images, train_labels

  '''
  im2show = validation_images[500, :, :, :]
  imgplot = plt.imshow(numpy.squeeze(im2show))
  show()
  '''

  return base.Datasets(train=train, validation=validation, test=test)


def load_SVHN(filename):
  data_dir = "./datasets/svhn"
  dset = scipy.io.loadmat(os.path.join(data_dir,filename))
  X = dset['X'].astype(numpy.float32)
  y = numpy.subtract(dset['y'], 1)

  '''
  im2show = numpy.transpose(X, (3, 0, 1, 2))[500,:,:,:]
  imgplot = plt.imshow(numpy.squeeze(im2show))
  show()

  '''

  if X.shape[3]>100000:
    X = X[:,:,:,0:100000]
    y = y[0:100000]
  elif X.shape[3]>60000:
    X = X[:,:,:,0:60000]
    y = y[0:60000]


  X = numpy.multiply(X, 1.0 / 255.0)
  #seed = 547
  #numpy.random.seed(seed)
  #numpy.random.shuffle(X)
  #numpy.random.seed(seed)
  #numpy.random.shuffle(y)
  y_dim = 10

  y_vec = numpy.zeros((len(y), y_dim), dtype=numpy.float)
  for i, label in enumerate(y):
      y_vec[i, y[i]] = 1.0

  '''
  im2show = numpy.transpose(X, (3, 0, 1, 2))[500, :, :, :]
  imgplot = plt.imshow(numpy.squeeze(im2show))
  show()
  '''

  return numpy.transpose(X, (3, 0, 1, 2)), y_vec
  #return numpy.transpose(X, (3, 2, 0, 1)), y_vec
