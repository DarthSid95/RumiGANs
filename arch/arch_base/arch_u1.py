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


class ARCH_u1_dead():
	def __init__(self):
		print("Creating 1-D Uniform architectures for base cases ")
		return

	def generator_model_u1(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(20, use_bias=True, input_shape=(self.noise_dims,),kernel_initializer=init_fn))
		model.add(layers.ReLU())

		model.add(layers.Dense(30, use_bias=True, kernel_initializer=init_fn))
		model.add(layers.ReLU())

		model.add(layers.Dense(15, use_bias=True, kernel_initializer=init_fn))
		model.add(layers.ReLU())

		model.add(layers.Dense(8, use_bias=True, kernel_initializer=init_fn))
		model.add(layers.ReLU())

		model.add(layers.Dense(1, use_bias=True, kernel_initializer=init_fn))
		model.add(layers.ReLU())

		return model


	def discriminator_model_u1(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(20, use_bias=False, input_shape=(1,), kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(20, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(10, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1, use_bias = False))
		# model.add(layers.Softmax())

		return model

	def show_result_u1(self, images = None, num_epoch = 0, show = False, save = False, path = 'result.png'):
		fig = plt.figure()
		ax = fig.add_subplot(111)
		# basis = np.expand_dims(np.linspace(-10., 10., int(1e4), dtype=np.float32), axis = 1)
		# disc = self.discriminator(basis,training= False)
		# disc /= max(abs(disc))
		print("Gaussian Stats : True mean {} True Sigma {}, Fake mean {} Fake Sigma {}".format(np.mean(self.reals), np.std(self.reals), np.mean(images), np.std(images) ))
		if self.res_flag:
			self.res_file.write("Gaussian Stats : True mean {} True Sigma {} \n Fake mean {} Fake Sigma {}".format(np.mean(self.reals), np.cov(self.reals,rowvar = False), np.mean(images), np.cov(images,rowvar = False) ))

		plt.rcParams.update({
			"pgf.texsystem": "pdflatex",
			"font.family": "serif",  # use serif/main font for text elements
			"font.size":10,
			"font.serif": [], 
			"text.usetex": True,     # use inline math for ticks
			"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			"pgf.preamble": [
				 r"\usepackage[utf8x]{inputenc}",
				 r"\usepackage[T1]{fontenc}",
				 r"\usepackage{cmbright}",
				 ]
		})

		with PdfPages(path+'_Classifier.pdf', metadata={'author': 'Siddarth Asokan'}) as pdf:
			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			# ax1.scatter(self.reals[:10000], np.ones_like(self.reals[:10000]), c='r',label='Real Data')
			ax1.scatter(self.train_data[self.org_labels == 0][0:1000], 1*np.ones_like(self.train_data[self.org_labels == 0])[0:1000], c='r', label='Real Data0')
			ax1.scatter(self.train_data[self.org_labels == 1][0:1000], 0.3*np.ones_like(self.train_data[self.org_labels == 1])[0:1000], c='b', label='Real Data1')
			ax1.scatter(self.train_data[self.org_labels == 2][0:1000], 5*np.ones_like(self.train_data[self.org_labels == 2])[0:1000], c='g', label='Real Data2')
			ax1.scatter(self.train_data[self.org_labels == 3][0:1000], 0.3*np.ones_like(self.train_data[self.org_labels == 3])[0:1000], c='m', label='Real Data3')
			ax1.scatter(self.train_data[self.org_labels == 4][0:1000], 1*np.ones_like(self.train_data[self.org_labels == 4])[0:1000], c='k',label='Real Data4')
			ax1.scatter(images[:10000], -0.5*np.ones_like(images[:10000]), c='g',label='Fake Data')
			# ax1.plot(basis, disc, c='b',label='Discriminator')
			ax1.legend(loc = 'upper right')

			label = 'Epoch {0}'.format(num_epoch)
			fig1.text(0.5, 0.04, label, ha='center')

			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)
