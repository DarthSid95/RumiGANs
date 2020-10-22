from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json
import glob
from sklearn.manifold import TSNE
# import prd_score as prd

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
# from matplotlib.backends.backend_pgf import PdfPages
# from gan_data import *
# FLAGS(sys.argv)
# tf.keras.backend.set_floatx('float64')

class ARCH_gmm2():
	def __init__(self):
		print("Creating 1-D GMM architectures for base cases ")
		return

	def generator_model_gmm2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(10, kernel_initializer=init_fn, use_bias = True)(inputs)
		enc1 = tf.keras.layers.ReLU()(enc1)

		enc2 = tf.keras.layers.Dense(20, kernel_initializer=init_fn, use_bias = True)(enc1)
		enc2 = tf.keras.layers.ReLU()(enc2)

		enc3 = tf.keras.layers.Dense(4, kernel_initializer=init_fn, use_bias = True)(enc2)
		# enc3 = tf.keras.layers.ReLU()(enc3)

		enc4 = tf.keras.layers.Dense(1, kernel_initializer=init_fn, use_bias = True)(enc3)
		# enc4 = tf.keras.layers.ReLU(max_value = 1.)(enc4)

		model = tf.keras.Model(inputs = inputs, outputs = enc4)

		return model

	def discriminator_model_gmm2(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)


		model = tf.keras.Sequential()
		model.add(layers.Dense(256, use_bias=False, input_shape=(1,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())


		model.add(layers.Dense(64, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		# model.add(layers.Dense(32, use_bias=False, kernel_initializer=init_fn))
		# # model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())

		model.add(layers.Dense(8, use_bias=False, kernel_initializer=init_fn))
		# # model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(2, use_bias=False ,kernel_initializer=init_fn))

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())

		return model

	def show_result_gmm2(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = True):

		# Define a single scalar Normal distribution
		basis = np.expand_dims(np.array(np.arange(self.MIN*10,self.MAX*10,1)/10.0),axis=1)
		disc = self.discriminator(basis,training = False)
		disc /= max(abs(disc))

		if self.paper and (self.total_count.numpy() == 1 or self.total_count.numpy()%1000 == 0):
			np.save(path+'_disc.npy',np.array(disc))
			np.save(path+'_reals.npy',np.array(self.reals))
			np.save(path+'_fakes.npy',np.array(self.fakes))
		
		if self.colab == 1:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import FigureCanvasPgf
			matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "serif",  # use serif/main font for text elements
				"font.size":10,	
				"font.serif": [], 
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
				# "pgf.preamble": [
				# 	 r"\usepackage[utf8x]{inputenc}",
				# 	 r"\usepackage[T1]{fontenc}",
				# 	 r"\usepackage{cmbright}",
				# 	 ]
			})

		with PdfPages(path+'_Classifier.pdf', metadata={'author': 'Siddarth Asokan'}) as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=-0.5,top=1.8)
			ax1.scatter(self.reals, np.zeros_like(self.reals), c='r', linewidth = 1.5, label='Real Data', marker = '.')
			ax1.scatter(images, np.zeros_like(images), c='g', linewidth = 1.5, label='Fake Data', marker = '.')
			ax1.plot(basis,disc, c='b', linewidth = 1.5, label='Discriminator')
			ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)
