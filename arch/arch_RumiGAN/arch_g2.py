from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
from matplotlib.backends.backend_pgf import PdfPages


class ARCH_g2():
	def __init__(self):
		print("Initializing base G2 architectures for SubGAN")
		return

	def generator_model_g2(self):
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(50, use_bias=True, input_shape=(self.noise_dims,), kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())
		model.add(layers.Dense(100, use_bias=True, kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())
		# model.add(layers.Dense(70, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.LeakyReLU())
		model.add(layers.Dense(10, use_bias=True,kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())
		model.add(layers.Dense(2, use_bias=True, activation = 'tanh',kernel_initializer=init_fn))
		# model.add(layers.LeakyReLU())
		return model

	def discriminator_model_g2(self):
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(10, use_bias=False, input_shape=(2,), kernel_initializer=init_fn, activation = 'tanh'))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(20, use_bias=False, kernel_initializer=init_fn))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(5, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())

		return model

	def show_result_g2(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = True):

		# Define a single scalar Normal distribution.

		basis = np.expand_dims(np.linspace(-10., 10., int(1e4), dtype=np.float32), axis = 1)

		if self.paper and (self.total_count.numpy() == 1 or self.total_count.numpy()%1000 == 0):
			np.save(path+'_reals_pos.npy',np.array(self.reals_pos))
			np.save(path+'_reals_neg.npy',np.array(self.reals_neg))
			np.save(path+'_fakes.npy',np.array(self.fakes))
		
		plt.rcParams.update({
			"pgf.texsystem": "pdflatex",
			"font.family": "serif",  # use serif/main font for text elements
			"font.size":12,
			"font.serif": [], 
			"text.usetex": True,     # use inline math for ticks
			"pgf.rcfonts": False,    # don't setup fonts from rc parameters
		})

		with PdfPages(path+'_Classifier.pdf', metadata={'author': 'Siddarth Asokan'}) as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=self.MIN,top=self.MAX)
			ax1.scatter(self.reals_pos[:,0], self.reals_pos[:,1], c='r', linewidth = 1.5, label='Positive Class Data', marker = '.')
			ax1.scatter(self.reals_neg[:,0], self.reals_neg[:,1], c='b', linewidth = 1.5, label='Negative Class Data', marker = '.')
			ax1.scatter(images[:,0], images[:,1], c='g', linewidth = 1.5, label='Fake Data', marker = '.')
			ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)