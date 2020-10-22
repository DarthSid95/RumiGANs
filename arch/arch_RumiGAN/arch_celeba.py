from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from absl import flags
FLAGS = flags.FLAGS


class ARCH_celeba():
	def __init__(self):
		print("Creating CelebA architectures for SubGAN case")
		return

	def generator_model_celeba(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)#
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape = (self.noise_dims,))

		dec1 = tf.keras.layers.Dense(int(self.output_size/16)*int(self.output_size/16)*1024, kernel_initializer=init_fn, use_bias=False)(inputs)		
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		un_flat = tf.keras.layers.Reshape([int(self.output_size/16),int(self.output_size/16),1024])(dec1) #4x4x1024

		deconv1 = tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(un_flat) #8x8x512 , New is 512
		deconv1 = tf.keras.layers.BatchNormalization()(deconv1)
		deconv1 = tf.keras.layers.LeakyReLU()(deconv1)

		deconv2 = tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(deconv1) #16x16x256 , New is 512
		deconv2 = tf.keras.layers.BatchNormalization()(deconv2)
		deconv2 = tf.keras.layers.LeakyReLU()(deconv2)

		deconv4 = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(deconv2) #32x32x128 , New is 1024
		deconv4 = tf.keras.layers.BatchNormalization()(deconv4)
		deconv4 = tf.keras.layers.LeakyReLU()(deconv4)

		out = tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn, activation = 'sigmoid')(deconv4) #64x64x3

		model = tf.keras.Model(inputs = inputs, outputs = out)
		return model

	def discriminator_model_celeba(self):
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
		if self.gan == 'SGAN':
			model.add(layers.Activation( activation = 'sigmoid'))

		return model

	### NEED TO FIX WITH SELF VARS
	def show_result_celeba(self, images = None, num_epoch = 0, show = False, save = False, path = 'result.png'):

		import logging
		logger = logging.getLogger()
		old_level = logger.level
		logger.setLevel(100)

		images = tf.reshape(images, [images.shape[0],self.output_size,self.output_size,3])
		images_on_grid = self.image_grid(input_tensor = images, grid_shape = (self.num_to_print,self.num_to_print),image_shape=(self.output_size,self.output_size),num_channels=3)
		fig = plt.figure(figsize=(7,7))
		ax1 = fig.add_subplot(111)
		ax1.cla()
		ax1.axis("off")
		ax1.imshow(np.clip(images_on_grid,0.,1.))

		label = 'Epoch {0}'.format(num_epoch)
		plt.title(label, fontsize=8)
		if save:
			plt.tight_layout()
			plt.savefig(path)
		if show:
			plt.show()
		else:
			plt.close()

		# size_figure_grid = 5
		# fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
		# for i in range(size_figure_grid):
		# 	for j in range(size_figure_grid):
		# 		ax[i, j].get_xaxis().set_visible(False)
		# 		ax[i, j].get_yaxis().set_visible(False)
				
		# images = images.numpy()
		# for k in range(size_figure_grid*size_figure_grid):
		# 	i = k // size_figure_grid
		# 	j = k % size_figure_grid
		# 	ax[i, j].cla()
		# 	ax[i, j].imshow(images[k], cmap='gray')

		# label = 'Epoch {0}'.format(num_epoch)
		# fig.text(0.5, 0.04, label, ha='center')

		# if save:
		# 	plt.savefig(path)

		# if show:
		# 	plt.show()
		# else:
		# 	plt.close()
		logger.setLevel(old_level)


	def CelebA_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(80,80,3), classes=1000)

	def FID_celeba(self):

		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[80,80])
				# This will convert to float values in [0, 1]
				image = tf.subtract(image,127.5)
				image = tf.divide(image,127.5)
				# image = tf.scalar_mul(2.0,image)
				
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1
			if self.testcase in ['bald', 'hat']:
				self.fid_train_images_names = self.fid_train_images
			else:
				random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
				print(random_points)
				self.fid_train_images_names = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images_names)
			self.fid_image_dataset = self.fid_image_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

			# for element in self.fid_image_dataset:
			# 	self.fid_images = element
			# 	break

			self.CelebA_Classifier()


		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:
				# print(self.fid_train_images.shape)
				noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator(noise, training=False)
				# preds = preds[:,:,:].numpy()		
				preds = tf.image.resize(preds, [80,80])
				preds = tf.scalar_mul(2.,preds)
				preds = tf.subtract(preds,1.0)
				preds = preds.numpy()

				act1 = self.FID_model.predict(image_batch)
				act2 = self.FID_model.predict(preds)
				try:
					self.act1 = np.concatenate((self.act1,act1), axis = 0)
					self.act2 = np.concatenate((self.act2,act2), axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2

			self.eval_FID()
			return
	###-----------------
	# def CelebA_Classifier(self):
	# 	self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(75,75,3), classes=1000)

	# def FID_celeba(self):

	# 	def data_reader_faces(filename):
	# 		with tf.device('/CPU'):
	# 			print(tf.cast(filename[0],dtype=tf.string))
	# 			image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
	# 			# Don't use tf.image.decode_image, or the output shape will be undefined
	# 			image = tf.image.decode_jpeg(image_string, channels=3)
	# 			image.set_shape([218,178,3])
	# 			image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
	# 			image = tf.image.resize(image,[75,75])
	# 			# This will convert to float values in [0, 1]
	# 			image = tf.divide(image,255.0)
	# 			image = tf.scalar_mul(2.0,image)
	# 			image = tf.subtract(image,1.0)
	# 			# image = tf.image.convert_image_dtype(image, tf.float16)
	# 		return image

	# 	if self.FID_load_flag == 0:
	# 		### First time FID call setup
	# 		self.FID_load_flag = 1	
	# 		random_points = tf.keras.backend.random_uniform([15000], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
	# 		print(random_points)
	# 		self.fid_train_images_names = self.fid_train_images[random_points]

	# 		## self.fid_train_images has the names to be read. Make a dataset with it
	# 		self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images_names)
	# 		self.fid_image_dataset = self.fid_image_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
	# 		self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

	# 		# for element in self.fid_image_dataset:
	# 		# 	self.fid_images = element
	# 		# 	break

	# 		self.CelebA_Classifier()


	# 	if self.mode == 'fid':
	# 		print(self.checkpoint_dir)
	# 		# print('logs/130919_ELeGANt_mnist_lsgan_base_01/130919_ELeGANt_mnist_lsgan_base_Results_checkpoints')
	# 		self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
	# 		print('Models Loaded Successfully')

	# 	with tf.device(self.device):
	# 		for image_batch in self.fid_image_dataset:
	# 			# print(self.fid_train_images.shape)
	# 			noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
	# 			preds = self.generator(noise, training=False)
	# 			# preds = preds[:,:,:].numpy()		
	# 			preds = tf.image.resize(preds, [75,75])
	# 			preds = tf.scalar_mul(2.,preds)
	# 			preds = tf.subtract(preds,1.0)
	# 			preds = preds.numpy()

	# 			act1 = self.FID_model.predict(image_batch)
	# 			act2 = self.FID_model.predict(preds)
	# 			try:
	# 				self.act1 = np.concatenate((self.act1,act1), axis = 0)
	# 				self.act2 = np.concatenate((self.act2,act2), axis = 0)
	# 			except:
	# 				self.act1 = act1
	# 				self.act2 = act2
	# 		self.eval_FID()
	# 		return

	####------------------------

	# def CelebA_Classifier(self):
	# 	self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(80,80,3), classes=1000)

	# def FID_celeba(self):
	# 	if self.FID_load_flag == 0:
	# 		### First time FID call setup
	# 		self.FID_load_flag = 1	
	# 		random_points = tf.keras.backend.random_uniform([1000], minval=0, maxval=int(self.train_data_pos.shape[0]), dtype='int32', seed=None)
	# 		print(random_points)
	# 		self.fid_train_images = self.fid_train_images[random_points]
	# 		self.fid_train_images = tf.image.resize(self.fid_train_images, [80,80])
	# 		self.fid_train_images = self.fid_train_images.numpy()

	# 		self.CelebA_Classifier()


	# 	if self.mode == 'fid':
	# 		print(self.checkpoint_dir)
	# 		# print('logs/130919_ELeGANt_mnist_lsgan_base_01/130919_ELeGANt_mnist_lsgan_base_Results_checkpoints')
	# 		self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
	# 		print('Models Loaded Successfully')

	# 	with tf.device(self.device):
	# 		# print(self.fid_train_images.shape)
	# 		preds = self.generator(tf.random.normal([1000, self.noise_dims]), training=False)
	# 		# preds = preds[:,:,:].numpy()		
	# 		preds = tf.image.resize(preds, [80,80])
	# 		preds = preds.numpy()

	# 		self.act1 = self.FID_model.predict(self.fid_train_images)
	# 		self.act2 = self.FID_model.predict(preds)
	# 		self.eval_FID()
	# 		return


		# if self.FID_load_flag == 0:
		# 	### First time FID call setup
		# 	self.FID_load_flag = 1	
		# 	random_points = tf.keras.backend.random_uniform([1000], minval=0, maxval=int(self.train_data_pos.shape[0]), dtype='int32', seed=None)
		# 	print(random_points)
		# 	fid_train_names = self.train_data_pos[random_points]

		# 	def data_reader_faces(filename):
		# 		with tf.device('/CPU'):
		# 			# print(tf.cast(filename[0],dtype=tf.string))
		# 			image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
		# 			# Don't use tf.image.decode_image, or the output shape will be undefined
		# 			image = tf.image.decode_jpeg(image_string, channels=3)
		# 			image.set_shape([218,178,3])
		# 			image = tf.image.resize(image,[80,80])

		# 			# This will convert to float values in [0, 1]
		# 			image = tf.divide(image,255.0)
		# 			# image = tf.image.convert_image_dtype(image, tf.float32)
		# 			# image = tf.divide(image,255.0)
		# 		return image

		# 	self.fid_dataset = tf.data.Dataset.from_tensor_slices((fid_train_names))
		# 	if not FLAGS.colab:
		# 		self.fid_dataset = self.fid_dataset.map(data_reader_faces, num_parallel_calls=int(self.num_parallel_calls))
		# 	self.fid_dataset = self.fid_dataset.batch(1000)

		# 	self.CelebA_Classifier()


		# if self.mode == 'fid':
		# 	print(self.checkpoint_dir)
		# 	# print('logs/130919_ELeGANt_mnist_lsgan_base_01/130919_ELeGANt_mnist_lsgan_base_Results_checkpoints')
		# 	self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
		# 	print('Models Loaded Successfully')

		# with tf.device(self.device):
		# 	# print(self.fid_train_images.shape)
		# 	preds = self.generator(tf.random.normal([1000, self.noise_dims]), training=False)
		# 	# preds = preds[:,:,:].numpy()		
		# 	preds = tf.image.resize(preds, [80,80])
		# 	preds = preds.numpy()

		# 	# calculate latent representations
		# 	for image_batch in self.fid_dataset:
		# 		image_batch = tf.image.resize(image_batch, [80,80])
		# 		image_batch = image_batch.numpy()
		# 		self.act1 = self.FID_model.predict(image_batch)
		# 		self.act2 = self.FID_model.predict(preds)
		# 		self.eval_FID()
		# 		return

