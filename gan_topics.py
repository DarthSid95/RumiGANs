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

from gan_data import *
from gan_arch import *

import tensorflow_probability as tfp
tfd = tfp.distributions
from matplotlib.backends.backend_pgf import PdfPages

# FLAGS(sys.argv)
# tf.keras.backend.set_floatx('float64')

'''
GAN_topic is the Overarching class file, where corresponding parents are instantialized, along with setting up the calling functions for these and files and folders for resutls, etc. data reading is also done from here. Sometimes display functions, architectures, etc may be modified here if needed (overloading parent classes)
'''

'''***********************************************************************************
********** GAN Baseline setup ********************************************************
***********************************************************************************'''
class GAN_Base(GAN_ARCH, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''

		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')


	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'


		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Batch Size * Dloop {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.batch_size_big, self.num_batches,self.print_step, self.save_step))


	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data = eval(self.gen_func)

		self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)

		self.train_dataset_size = self.train_data.shape[0]



'''***********************************************************************************
********** AConditional GAN (cGAN-PD, ACGAN, TACGAN) setup ****************************
***********************************************************************************'''
class GAN_CondGAN(GAN_ARCH, GAN_DATA_CondGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''

		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_CondGAN.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')


	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.train_labels, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		if self.loss == 'FS':
			self.gen_model = 'self.generator_model_'+self.data+'_'+self.latent_kind+'()'
			self.disc_model = 'self.discriminator_model_'+self.data+'_'+self.latent_kind+'()' 
			self.EncDec_func = 'self.encoder_model_'+self.data+'_'+self.latent_kind+'()'
			self.DEQ_func = 'self.discriminator_ODE()'

		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Batch Size * Dloop {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.batch_size_big, self.num_batches,self.print_step, self.save_step))


	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data, self.train_labels = eval(self.gen_func)

		self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.num_batches = int(np.floor((self.train_data.shape[0])/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		print("Dataset created - this is it")
		print(self.train_dataset)

		self.train_dataset_size = self.train_data.shape[0]

	def get_noise(self,noise_case,batch_size):
		noise = tf.random.normal([batch_size, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)
		if noise_case == 'test':
			if self.data in ['mnist', 'cifar10']:
				if self.testcase in ['single', 'few']:
					noise_labels = self.number*np.ones((batch_size,1)).astype('int32')
				elif self.testcase in ['sharp']:
					noise_labels = np.expand_dims(np.random.choice([1,2,4,5,7,9], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['even']:
					noise_labels = np.expand_dims(np.random.choice([0,2,4,6,8], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['odd']:
					noise_labels = np.expand_dims(np.random.choice([1,3,5,7,9], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['animals']:
					noise_labels = np.expand_dims(np.random.choice([2,3,4,5,6,7], batch_size), axis = 1).astype('int32')
			elif self.data in ['celeba']:
				if self.testcase in ['male', 'fewmale', 'bald', 'hat']:
					noise_labels = np.ones((batch_size,1)).astype('int32')
				elif self.testcase in ['female', 'fewfemale']:
					noise_labels = np.zeros((batch_size,1)).astype('int32')
		if noise_case == 'train':
			noise_labels = np.random.randint(0, self.num_classes, batch_size)
			if self.data == 'celeba':
				noise_labels = np.expand_dims(noise_labels, axis = 1)

		return noise, noise_labels



'''***********************************************************************************
********** GAN RumiGAN setup *********************************************************
***********************************************************************************'''
class GAN_RumiGAN(GAN_ARCH, GAN_DATA_RumiGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''
		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_RumiGAN.__init__(self)


	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data_pos, self.train_data_neg, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Batch Size * Dloop {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.batch_size_big, self.num_batches,self.print_step, self.save_step))

	def get_data(self):
		
		with tf.device('/CPU'):
			self.train_data_pos, self.train_data_neg = eval(self.gen_func)

			if self.data in [ 'wce', 'kid']:
				self.max_data_size = 4*max(self.train_data_pos.shape[0],self.train_data_neg.shape[0])
			else:
				self.max_data_size = max(self.train_data_pos.shape[0],self.train_data_neg.shape[0])

			self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
			self.num_batches = int(np.floor(self.max_data_size/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset_pos, self.train_dataset_neg = eval(self.dataset_func)

			self.train_dataset_size = self.max_data_size


'''***********************************************************************************
********** GAN AAE setup *************************************************************
***********************************************************************************'''
class GAN_WAE(GAN_ARCH, GAN_DATA_WAE):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''

		# self.lr_AE_Enc = FLAGS_dict['lr_AE_Enc']
		# self.lr_AE_Dec = FLAGS_dict['lr_AE_Dec']
		# self.AE_count = FLAGS_dict['AE_count']

		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_AAE.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')

		self.noise_setup()

	def noise_setup(self):
		self.num_of_components = 20

		probs = list((1/self.num_of_components)*np.ones([self.num_of_components]))
		stddev_scale = list(0.8*np.ones([self.num_of_components]))
		# locs = list(np.random.uniform(size = [10, self.latent_dims], low = 1., high = 8.))
		locs = np.random.uniform(size = [self.num_of_components, self.latent_dims], low = -3., high = 3.)
		self.locs = tf.Variable(locs)
		locs = [list(x) for x in list(locs)]
		
		print(locs)#[[7.5, 5], [5, 7.5], [2.5,5], [5,2.5], [7.5*0.7071, 7.5*0.7071], [2.5*0.7071, 7.5*0.7071], [7.5*0.7071, 2.5*0.7071], [2.5*0.7071, 2.5*0.7071] ]
		# stddev_scale = [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5]

		# self.gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		# probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
		# locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.5*1.7071, 0.5*1.7071], [0.5*0.2929, 0.5*1.7071], [0.5*1.7071, 0.5*0.2929], [0.5*0.2929, 0.5*0.2929] ]
		# stddev_scale = [.04, .04, .04, .04, .04, .04, .04, .04]
		self.gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		self.gN = tfd.Normal(loc=1.25, scale=1.)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.encdec_model = 'self.encdec_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		if self.loss == 'FS':
			self.disc_model = 'self.discriminator_model_FS()' 
			self.DEQ_func = 'self.discriminator_ODE()'

		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Batch Size * Dloop {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.batch_size_big, self.num_batches,self.print_step, self.save_step))


	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data = eval(self.gen_func)

		self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		self.train_dataset_size = self.train_data.shape[0]


	def get_noise(self,batch_size):
		###Uncomment for the continues CelebaCode on Vega
		if self.noise_kind == 'gaussian_trunc':
			noise = tfp.distributions.TruncatedNormal(loc=0., scale=0.3, low=-1., high=1.).sample([batch_size, self.latent_dims])

		###Uncomment for the continues CIFAR10Code on Vayu
		if self.noise_kind == 'gmm':
			noise = self.gmm.sample(sample_shape=(int(batch_size.numpy())))

		if self.noise_kind == 'gN':
			noise = self.gN.sample(sample_shape=(int(batch_size.numpy()),self.latent_dims))

		# noise = tfp.distributions.TruncatedNormal(loc=1.25, scale=0.75, low=0., high=3.).sample([batch_size, self.latent_dims])

		# tf.random.normal([100, self.latent_dims], mean = self.locs.numpy()[i], stddev = 1.)
		if self.noise_kind == 'gaussian':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 1.0)

		if self.noise_kind == 'gaussian_s2':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = np.sqrt(2))

		if self.noise_kind == 'gaussian_1m1':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 0.25)

		if self.noise_kind == 'gaussian_05':
			noise = tfp.distributions.TruncatedNormal(loc=2.5, scale=1., low=0., high=5.).sample([batch_size, self.latent_dims])

		if self.noise_kind == 'gaussian_02':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.7*np.ones((1,self.latent_dims)), stddev = 0.2*np.ones((1,self.latent_dims)))

		if self.noise_kind == 'gaussian_01':
			noise = tfp.distributions.TruncatedNormal(loc=0.5, scale=0.2, low=0., high=1.).sample([batch_size, self.latent_dims])



		return noise

