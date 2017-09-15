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

import matplotlib.pyplot as plt
from pylab import show

import scipy.io
import scipy.misc
import os
import numpy
import pickle
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def load_mnist(filename='mnist32.npz', data_dir = './datasets/mnist'):

  mnistdata = numpy.load(os.path.join(data_dir, filename))
  train_images = mnistdata['train_image']
  ytrain_labels = mnistdata['train_label']
  test_images = mnistdata['test_image']
  ytest_labels = mnistdata['test_label']

  '''
  seed = 547
  numpy.random.seed(seed)
  numpy.random.shuffle(X)
  numpy.random.seed(seed)
  numpy.random.shuffle(y)
  '''
  y_dim = 10

  train_labels = numpy.zeros((len(ytrain_labels), y_dim), dtype=numpy.float)
  for i, label in enumerate(ytrain_labels):
      train_labels[i, ytrain_labels[i]] = 1.0

  test_labels = numpy.zeros((len(ytest_labels), y_dim), dtype=numpy.float)
  for i, label in enumerate(ytest_labels):
      test_labels[i, ytest_labels[i]] = 1.0

  train_images = numpy.transpose(train_images, (0, 2, 3, 1))
  test_images = numpy.transpose(test_images, (0, 2, 3, 1))

  return train_images,train_labels,test_images,test_labels


  #return numpy.transpose(X, (3, 2, 0, 1)), y_vec



class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               pernum=10):
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
      '''if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
      '''
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
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
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

  def get_labeled(self):
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

      #print (self.llabels[start])
      #im2show = self.limages[start]
      #imgplot = plt.imshow(numpy.squeeze(im2show))
      #show()
      return self.limages[start:end], self.llabels[start:end]



def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=False,
                   validation_size=5000,
                   filename= 'mnist32.npz',
                   color=False,
                   noise=False,
                   lpernum=10):
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  data_dir = "./data/mnist"

  if noise:
    pickle_file = "./datasets/mnist/picklednoisymnist.pickle"
  else:
    pickle_file = "./datasets/mnist/pickledmnist.pickle"
  if not os.path.isfile(pickle_file):
    print("mnist data not found, pickling datasets")

    train_images, train_labels, test_images, test_labels = load_mnist()

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    if (color):
      train_images = numpy.concatenate((train_images, train_images, train_images), axis=3)
      test_images = numpy.concatenate((test_images, test_images, test_images), axis=3)
      validation_images = numpy.concatenate((validation_images, validation_images, validation_images), axis=3)

    if noise:
      for i in range(train_images.shape[0]):
        sign = numpy.random.randint(0, 2) * 2 - 1
        if sign < 0:
          x = numpy.ones((1, train_images.shape[1],train_images.shape[2],train_images.shape[3]))*255 - train_images[i, :, :, :]
          train_images[i, :, :, :] = (x/255) * 2 - 1
        else:
          train_images[i, :, :, :] = (train_images[i, :, :, :] / 255) * 2 - 1

        #im2show = train_images[i, :, :, :]
        #imgplot = plt.imshow(numpy.squeeze(im2show))
        #show()

      for i in range(validation_images.shape[0]):
        sign = numpy.random.randint(0, 2) * 2 - 1
        if sign < 0:
          x = numpy.ones((1, validation_images.shape[1],validation_images.shape[2],validation_images.shape[3]))*255 - validation_images[i, :, :, :]
          validation_images[i, :, :, :] = (x/255) * 2 - 1
        else:
          validation_images[i, :, :, :] = (validation_images[i, :, :, :] / 255) * 2 - 1

      for i in range(test_images.shape[0]):
        sign = numpy.random.randint(0, 2) * 2 - 1
        if sign < 0:
          x = numpy.ones((1, test_images.shape[1],test_images.shape[2],test_images.shape[3]))*255 - test_images[i, :, :, :]
          test_images[i, :, :, :] = (x/255) * 2 - 1
        else:
          test_images[i, :, :, :] = (test_images[i, :, :, :] / 255) * 2 - 1

    else:
      for c in range(train_images.shape[3]):
        for i in range(train_images.shape[0]):
          x = (train_images[i, :, :, c])
          # train_images[i, :, :, 0] = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)) * 2 - 1
          train_images[i, :, :, c] = (x / 255) * 2 - 1
          # train_images[i, :, :, 0] = x / 255

        for i in range(validation_images.shape[0]):
          x = (validation_images[i, :, :, c])
          # validation_images[i, :, :, 0] = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)) * 2 - 1
          validation_images[i, :, :, c] = (x / 255) * 2 - 1
          # validation_images[i, :, :, 0] = x / 255

        for i in range(test_images.shape[0]):
          x = (test_images[i, :, :, c])
          # test_images[i, :, :, 0] = (x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)) * 2 - 1
          test_images[i, :, :, c] = (x / 255) * 2 - 1
          # test_images[i, :, :, 0] = x / 255

    '''if noise:
      for i in range(train_images.shape[0]):
        #xscale = int(32 * (numpy.random.random_sample() * 2.1 + .4))
        #yscale = int(32 * (numpy.random.random_sample() * .83 + .67))

        xscale = int(32 * (numpy.random.random_sample() * .8 + .6))
        yscale = int(32 * (numpy.random.random_sample() * .6 + .7))

        scale = numpy.random.uniform(.6, 1.5)
        sign = numpy.random.randint(0, 2) * 2 - 1
        scale = scale * sign

        offset = numpy.random.uniform(-.5, .5)

        paddedx = xscale + 32
        paddedy = yscale + 32

        newim = numpy.ones((1, paddedy, paddedx, 3)) / 50

        for c in range(train_images.shape[3]):
          x = (train_images[i, :, :, c])

          train_images[i, :, :, c] = ((x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)))
          # train_images[i, :, :, c] = (x / 255) * 2 - 1
          # train_images[i, :, :, 0] = x / 255
          # s = ((x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)))* 2
        # train_images[i, :, :, :] = scipy.misc.imresize(scipy.misc.imresize(train_images[i, :, :, :],[1,int(32*xscale),int(32*yscale),3]),[1,32,32,3])

        temp1 = scipy.misc.imresize(train_images[i, :, :, :], (int(yscale), int(xscale), 3)) / 255
        newim[0, int(paddedy / 2) - int(yscale / 2): int(paddedy / 2) - int(yscale / 2) + yscale,
        int(paddedx / 2) - int(xscale / 2): int(paddedx / 2) - int(xscale / 2) + xscale, :] = temp1
        newim = newim[:, int(paddedy / 2) - 16: int(paddedy / 2) + 16, int(paddedx / 2) - 16: int(paddedx / 2) + 16, :]
        temp2 = newim * scale + offset
        train_images[i, :, :, :] = ((temp2 + 2) / 4)
        # train_images[i, :, :, :] = temp2

        # im2show = train_images[i, :, :, :]
        # imgplot = plt.imshow(numpy.squeeze(im2show))
        # show()

      for i in range(validation_images.shape[0]):
        xscale = int(32 * (numpy.random.random_sample() * .8 + .6))
        yscale = int(32 * (numpy.random.random_sample() * .6 + .7))

        scale = numpy.random.uniform(.6, 1.5)
        sign = numpy.random.randint(0, 2) * 2 - 1
        scale = scale * sign

        offset = numpy.random.uniform(-.5, .5)

        paddedx = xscale + 32
        paddedy = yscale + 32

        newim = numpy.ones((1, paddedy, paddedx, 3)) / 50

        for c in range(validation_images.shape[3]):
          x = (validation_images[i, :, :, c])

          validation_images[i, :, :, c] = ((x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)))
          # train_images[i, :, :, c] = (x / 255) * 2 - 1
          # train_images[i, :, :, 0] = x / 255
          # s = ((x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)))* 2
        # train_images[i, :, :, :] = scipy.misc.imresize(scipy.misc.imresize(train_images[i, :, :, :],[1,int(32*xscale),int(32*yscale),3]),[1,32,32,3])

        temp1 = scipy.misc.imresize(validation_images[i, :, :, :], (int(yscale), int(xscale), 3)) / 255
        newim[0, int(paddedy / 2) - int(yscale / 2): int(paddedy / 2) - int(yscale / 2) + yscale,
        int(paddedx / 2) - int(xscale / 2): int(paddedx / 2) - int(xscale / 2) + xscale, :] = temp1
        newim = newim[:, int(paddedy / 2) - 16: int(paddedy / 2) + 16, int(paddedx / 2) - 16: int(paddedx / 2) + 16, :]

        temp2 = newim * scale + offset
        validation_images[i, :, :, :] = ((temp2 + 2) / 4)

      for i in range(test_images.shape[0]):
        xscale = int(32 * (numpy.random.random_sample() * .8 + .6))
        yscale = int(32 * (numpy.random.random_sample() * .6 + .7))

        scale = numpy.random.uniform(.6, 1.5)
        sign = numpy.random.randint(0, 2) * 2 - 1
        scale = scale * sign

        offset = numpy.random.uniform(-.5, .5)

        paddedx = xscale + 32
        paddedy = yscale + 32

        newim = numpy.ones((1, paddedy, paddedx, 3)) / 50

        for c in range(test_images.shape[3]):
          x = (test_images[i, :, :, c])

          test_images[i, :, :, c] = ((x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)))
          # test_images[i, :, :, c] = (x / 255) * 2 - 1
          # test_images[i, :, :, 0] = x / 255
          # s = ((x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)))* 2
        # test_images[i, :, :, :] = scipy.misc.imresize(scipy.misc.imresize(test_images[i, :, :, :],[1,int(32*xscale),int(32*yscale),3]),[1,32,32,3])

        temp1 = scipy.misc.imresize(test_images[i, :, :, :], (int(yscale), int(xscale), 3)) / 255
        newim[0, int(paddedy / 2) - int(yscale / 2): int(paddedy / 2) - int(yscale / 2) + yscale,
        int(paddedx / 2) - int(xscale / 2): int(paddedx / 2) - int(xscale / 2) + xscale, :] = temp1
        newim = newim[:, int(paddedy / 2) - 16: int(paddedy / 2) + 16, int(paddedx / 2) - 16: int(paddedx / 2) + 16, :]

        temp2 = newim * scale + offset
        test_images[i, :, :, :] = ((temp2 + 2) / 4)'''




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

    print('Data pickled')
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
      print("loaded pickled mnist")








  #train_images = numpy.concatenate((train_images,train_images,train_images),1)
  #test_images = numpy.concatenate((test_images, test_images, test_images), 1)




  train = DataSet(train_images, train_labels,pernum=lpernum, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape,
                       pernum=lpernum)
  test = DataSet(test_images, test_labels, pernum=lpernum, dtype=dtype, reshape=reshape)

  #im2show = train_images[500, :, :, :]
  #imgplot = plt.imshow(numpy.squeeze(im2show))
  #show()

  return base.Datasets(train=train, validation=validation, test=test)

