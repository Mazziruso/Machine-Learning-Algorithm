import os
import sys
import numpy as np
import pickle
from PIL import Image
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed


# extract images from pkl
def extract_images(f):
	print("Extracting", f.name)
	face_imgs = pickle.load(f)
	num_images = 400
	rows = 57
	cols = 47
	data = face_imgs.reshape(-1, rows, cols, 1).astype(np.float32)
	return data


def dense_to_one_hot(labels_dense, num_class):
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_class
	labels_one_hot = np.zeros([num_labels, num_class])
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return labels_one_hot.astype(np.uint8)


def extract_labels(f, one_hot=False, num_class=40):
	print("Extracting", f.name)
	face_labels = pickle.load(f)
	num_labels = 400
	face_labels.astype(np.uint8)
	if one_hot:
		return dense_to_one_hot(face_labels, num_class)
	return face_labels


class DataSet(object):

	def __init__(self, imags, labels, one_hot=False, dtype=dtypes.float32, reshape=True, seed=None):
		seed1, seed2 = random_seed.get_seed(seed)
		np.random.seed(seed1 if seed is None else seed2)
		dtype = dtypes.as_dtype(dtype).base_dtype
		if dtype not in [dtypes.uint8, dtypes.float32]:
			raise TypeError('Invalid image data %r, expected uint8 or float32' % dtype)

		assert imags.shape[0]==labels.shape[0], ('images.shape: %s labels.shape: %s' % (imags.shape, labels.shape))
		self._num_examples = imags.shape[0]
		# convert shape from [num_imags, rows, cols, depth]
		# to [num_imags, rows*cols] (assuming depth==1)
		if reshape:
			assert imags.shape[3] == 1
			imags = imags.reshape(imags.shape[0], imags.shape[1]*imags.shape[2])
		if dtype == dtypes.float32:
			imags = imags.astype(np.float32)
			imags = np.multiply(imags, 1.0/255.0) # normalization
		self._images = imags
		self._labels = labels
		self._epochs_completed = 0  # 记录一次训练共用了几次整个数据集
		self._index_in_epoch = 0  # 记录当前训练的batch下标

	# property装饰器负责把一个方法变成一个属性，类似于set和get方法对类属性的保护
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

	def next_batch(self, batch_size, shuffle=True):
		# set the start of this batch
		start = self._index_in_epoch
		# shuffle for the first epoch
		if self._epochs_completed ==0 and start == 0 and shuffle:
			perm0 = np.arange(self.num_examples)
			np.random.shuffle(perm0)
			self._images = self.images[perm0]
			self._labels = self.labels[perm0]
		# get the next batch
		if start + batch_size > self.num_examples:
			# Finished the epoch
			self._epochs_completed += 1
			# get the rest examples in this epoch
			res_num_examples = self.num_examples - start
			images_rest_part = self._images[start:self.num_examples]
			labels_rest_part = self._labels[start:self.num_examples]
			# shuffle
			if shuffle:
				perm = np.arange(self._num_examples)
				np.random.shuffle(perm)
				self._images = self.images[perm]
				self._labels = self.labels[perm]
			# start next epoch
			start = 0
			self._index_in_epoch = batch_size - res_num_examples
			end = self._index_in_epoch
			images_new_part = self._images[start:end]
			labels_new_part = self._labels[start:end]
			return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
		else:
			# go to next batch
			self._index_in_epoch += batch_size
			end = self._index_in_epoch
			return self._images[start:end], self._labels[start:end]


def read_data_sets(one_hot=False, dtype=dtypes.float32, reshape=True, validation_size=40, seed=None):
	local_file = 'olivettifaces_images.pkl'
	with open(local_file, 'rb') as f:
		images = extract_images(f)

	local_file = 'olivettifaces_labels.pkl'
	with open(local_file, 'rb') as f:
		labels = extract_labels(f, one_hot=one_hot)

	# shuffle
	num_examples = images.shape[0]
	perm = np.arange(num_examples)
	np.random.shuffle(perm)
	images = images[perm]
	labels = labels[perm]

	train_images = np.empty((320, 57, 47, 1))
	train_labels = np.empty((320, 40))
	validation_images = np.empty((40, 57, 47, 1))
	validation_labels = np.empty((40, 40))
	test_images = np.empty((40, 57, 47, 1))
	test_labels = np.empty((40, 40))

	for i in range(40):
		train_images[i*8:i*8+8] = images[i*10:i*10+8]
		train_labels[i*8:i*8+8] = labels[i*10:i*10+8]
		validation_images[i] = images[i*10+8]
		validation_labels[i] = labels[i*10+8]
		test_images[i] = images[i*10+9]
		test_labels[i] = labels[i*10+9]

	train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
	validation = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape, seed=seed)
	test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)
	return base.Datasets(train=train, validation=validation, test=test)

