from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class ARCH_comma():
	def __init__(self):
		print("Creating comma.ai architectures for base cases ")
		return

	def generator_model_comma(self):

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape = (self.noise_dims,))

		dec1 = tf.keras.layers.Dense(int(self.output_size/16)*int(self.output_size/16)*1024, kernel_initializer=init_fn, use_bias=False)(inputs)		
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		un_flat = tf.keras.layers.Reshape([int(self.output_size/16),int(self.output_size/16),1024])(dec1) #4x4x1024

		deconv1 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(un_flat) #8x8x512
		deconv1 = tf.keras.layers.BatchNormalization()(deconv1)
		deconv1 = tf.keras.layers.ReLU()(deconv1)

		deconv2 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(deconv1) #16x16x256
		deconv2 = tf.keras.layers.BatchNormalization()(deconv2)
		deconv2 = tf.keras.layers.ReLU()(deconv2)

		deconv4 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(deconv2) #32x32x128
		deconv4 = tf.keras.layers.BatchNormalization()(deconv4)
		deconv4 = tf.keras.layers.ReLU()(deconv4)

		out = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn, activation = 'tanh')(deconv4) #64x64x3

		model = tf.keras.Model(inputs = inputs, outputs = out)
		return model

	def discriminator_model_comma(self):

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential() #64x64x3
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn,input_shape=[self.output_size, self.output_size, 3])) #32x32x64
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #16x16x128
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #8x8x256
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #4x4x512
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Flatten()) #8192x1
		model.add(layers.Dense(1)) #1x1

		return model

	### NEED TO FIX WITH SELF VARS
	def show_result_comma(self, images = None, num_epoch = 0, show = False, save = False, path = 'result.png'):
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

