from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



import tensorflow_probability as tfp
tfd = tfp.distributions


class ARCH_g1():
	def __init__(self):
		print("Creating 1-D Gaussian architectures for base cases ")
		return

	def generator_model_g1(self):
		init_fn_bias = tf.keras.initializers.glorot_uniform() #tf.keras.initializers.Identity()#
		init_fn_bias = tf.function(init_fn_bias, autograph=False)
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(1, use_bias=True, input_shape=(self.noise_dims,),kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		return model

	def discriminator_model_g1(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(5, use_bias=True, input_shape=(1,), kernel_initializer=init_fn))# bias_initializer = bias_init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(10, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(2, use_bias=True, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())

		model.add(layers.Dense(1, use_bias = True))
		# model.add(layers.Softmax())

		return model

	def show_result_g1(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = True):

		# Define a single scalar Normal distribution.
		pd_dist = tfd.Normal(loc=np.mean(self.reals), scale=np.std(self.reals))
		pg_dist = tfd.Normal(loc=np.mean(self.fakes), scale=np.std(self.fakes))

		# basis = np.expand_dims(np.linspace(-10., 10., int(1e4), dtype=np.float32), axis = 1)
		basis = np.expand_dims(np.linspace(self.MIN, self.MAX, int(1e4), dtype=np.float32), axis = 1)
		pd_vals = pd_dist.prob(basis)
		# pd_vals = pd_vals/max(pd_vals)
		pg_vals = pg_dist.prob(basis)
		# pg_vals = pg_vals/max(pg_vals)

		disc = self.discriminator(basis,training = False)
		disc = disc - min(disc)
		disc = (disc/max(disc))*1.0
		disc -= 0.50

		true_classifier = np.ones_like(basis)
		# true_classifier[pd_vals > pg_vals] = 0

		if self.paper and (self.total_count.numpy() == 1 or self.total_count.numpy() % 1000):
			np.save(path+'_disc_'+str(self.total_count.numpy())+'.npy',np.array(disc))
			np.save(path+'_reals_'+str(self.total_count.numpy())+'.npy',np.array(self.reals))
			np.save(path+'_fakes_'+str(self.total_count.numpy())+'.npy',np.array(self.fakes))
		
		if self.colab == 1:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
			
			# from matplotlib.backends.backend_pgf import FigureCanvasPgf
			# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
			# from matplotlib.backends.backend_pgf import PdfPages
			# plt.rcParams.update({
			# 	"pgf.texsystem": "pdflatex",
			# 	"font.family": "serif",  # use serif/main font for text elements
			# 	"font.size":10,	
			# 	"font.serif": [], 
			# 	"text.usetex": True,     # use inline math for ticks
			# 	"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			# 	# "pgf.preamble": [
			# 	# 	 r"\usepackage[utf8x]{inputenc}",
			# 	# 	 r"\usepackage[T1]{fontenc}",
			# 	# 	 r"\usepackage{cmbright}",
			# 	# 	 ]
			# })

		with PdfPages(path+'_Classifier.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=-0.5,top=1.8)
			ax1.plot(basis,pd_vals, linewidth = 1.5, c='r')
			ax1.plot(basis,pg_vals, linewidth = 1.5, c='g')
			ax1.scatter(self.reals, np.zeros_like(self.reals), c='r', linewidth = 1.5, label='Real Data', marker = '.')
			ax1.scatter(images, np.zeros_like(images), c='g', linewidth = 1.5, label='Fake Data', marker = '.')
			ax1.plot(basis,disc, c='b', linewidth = 1.5, label='Discriminator')
			if self.total_count < 20:
				ax1.plot(basis,true_classifier,'c--', linewidth = 1.5, label='True Classifier')
			ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)


			# if self.total_count > 10:
			# 	exit(0)