from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class ARCH_BSDS():
	def __init__(self):
		return
	def generator_model_BSDS(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()

		model.add(layers.Conv2D(128, (5, 5),input_shape=(self.output_size, self.output_size, 3, ), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		# print(model.output_shape)
		assert model.output_shape == (None, int(self.output_size/2), int(self.output_size/2), 128)
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/4), int(self.output_size/4), 64)
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/8), int(self.output_size/8), 64)
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/16), int(self.output_size/16), 32)
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())


		model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/8), int(self.output_size/8), 32)
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())


		model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/4), int(self.output_size/4), 64)
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/2), int(self.output_size/2), 128)
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn, activation='tanh'))
		assert model.output_shape == (None, self.output_size, self.output_size, 3)
		return model

	def discriminator_model_BSDS(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[self.output_size, self.output_size, 3,]))
		model.add(layers.LeakyReLU())
		# model.add(layers.Dropout(0.3))

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())
		# model.add(layers.Dropout(0.3))

		# model.add(layers.Flatten())
		model.add(layers.Dense(1))

		return model

	### NEED TO FIX WITH SELF VARS
	def show_result_BSDS(self, images = None, num_epoch = 0, show = False, save = False, path = 'result.png'):
		size_figure_grid = 5
		fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
		for i in range(size_figure_grid):
			for j in range(size_figure_grid):
				ax[i, j].get_xaxis().set_visible(False)
				ax[i, j].get_yaxis().set_visible(False)

		for k in range(size_figure_grid*size_figure_grid):
			i = k // size_figure_grid
			j = k % size_figure_grid
			ax[i, j].cla()
			ax[i, j].imshow(images[k], cmap='gray')

		label = 'Epoch {0}'.format(num_epoch)
		fig.text(0.5, 0.04, label, ha='center')

		if save:
			plt.savefig(path)

		if show:
			plt.show()
		else:
			plt.close()
