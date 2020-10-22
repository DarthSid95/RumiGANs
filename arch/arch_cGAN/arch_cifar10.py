from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from ops import *


class ARCH_cifar10():
	def __init__(self):
		print("Creating CIFAR-10 architectures for cGAN case")
		return

	def generator_model_cifar10(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		if self.label_style == 'base':

			noise_ip = tf.keras.Input(shape=(self.noise_dims, ))
			image_class = tf.keras.Input(shape=(self.num_classes,))
			gen_concat = tf.keras.layers.Concatenate()([noise_ip, image_class])
			gen_dense = layers.Dense(int(self.output_size/8)*int(self.output_size/8)*256)(gen_concat)
			gen_ip = layers.Reshape((int(self.output_size/8), int(self.output_size/8), 256))(gen_dense)

		elif self.label_style == 'embed':

			noise_ip = tf.keras.Input(shape=(self.noise_dims, ))
			image_class = tf.keras.Input(shape=(1,), dtype='int32')

			noise_den = layers.Dense(int(self.output_size/8)*int(self.output_size/8)*255, use_bias=False,kernel_initializer=init_fn)(noise_ip)
			noise_res =layers.Reshape((int(self.output_size/8), int(self.output_size/8),255))(noise_den)

			class_embed = tf.keras.layers.Embedding(input_dim = self.num_classes, output_dim = 10, embeddings_initializer='glorot_normal')(image_class)
			class_den = layers.Dense(int(self.output_size/8)*int(self.output_size/8), use_bias=False,kernel_initializer=init_fn)(class_embed)
			class_res = layers.Reshape((int(self.output_size/8), int(self.output_size/8), 1))(class_den)
			gen_ip = tf.keras.layers.Concatenate()([noise_res, class_res])

		elif self.label_style == 'multiply':

			noise_ip = tf.keras.Input(shape=(self.noise_dims, ))
			image_class = tf.keras.Input(shape=(1,), dtype='int32')

			class_embed = tf.keras.layers.Embedding(input_dim = self.num_classes, output_dim = self.noise_dims, embeddings_initializer='glorot_normal')(image_class)
			
			gen_multiply = tf.keras.layers.Multiply()([noise_ip,class_embed])
			gen_dense = layers.Dense(int(self.output_size/8)*int(self.output_size/8)*256)(gen_multiply)
			gen_ip = layers.Reshape((int(self.output_size/8), int(self.output_size/8), 256))(gen_dense)


		denc3 = tf.keras.layers.Conv2DTranspose(256, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(gen_ip) #4x4x256
		denc3 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc3)
		# denc3 = tf.keras.layers.Dropout(0.5)(denc3)
		denc3 = tf.keras.layers.LeakyReLU()(denc3)


		denc2 = tf.keras.layers.Conv2DTranspose(128, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(denc3) #8x8x128
		denc2 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc2)
		# denc2 = tf.keras.layers.Dropout(0.5)(denc2)
		denc2 = tf.keras.layers.LeakyReLU()(denc2)


		denc1 = tf.keras.layers.Conv2DTranspose(64, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(denc2) #16x16x64
		denc1 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc1)
		# denc1 = tf.keras.layers.Dropout(0.5)(denc1)
		denc1 = tf.keras.layers.LeakyReLU()(denc1)

		out = tf.keras.layers.Conv2DTranspose(3, 5,strides=1,padding='same', kernel_initializer=init_fn)(denc1) #32x32x3
		out =  tf.keras.layers.Activation( activation = 'tanh')(out)

		
		model = tf.keras.Model(inputs=[noise_ip, image_class], outputs=out)
		return model

	def discriminator_model_cifar10(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape = (self.output_size,self.output_size,3))

		conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 3])(inputs)
		conv1 = layers.BatchNormalization()(conv1)
		conv1 = layers.LeakyReLU()(conv1)
		# model.add(layers.Dropout(0.3))

		conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)(conv1)
		conv2 = layers.BatchNormalization()(conv2)
		conv2 = layers.LeakyReLU()(conv2)


		conv3 = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)(conv2)
		conv3 = layers.BatchNormalization()(conv3)
		conv3 = layers.LeakyReLU()(conv3)

		conv4 = layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same', kernel_initializer=init_fn)(conv3)
		conv4 = layers.BatchNormalization()(conv4)
		conv4 = layers.LeakyReLU()(conv4)

		flat = layers.Flatten()(conv3)
		dense1 = layers.Dense(50)(flat)

		dense2 = layers.Dense(1)(dense1)
	
		avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(conv4)

		if self.label_style == 'base':

			image_class = tf.keras.Input(shape=(self.num_classes,))
			den_pool = layers.Dense(10)(avg_pool)
			inner_prod = tf.keras.layers.dot([den_pool,image_class], axes = 1)

		if self.label_style in ['embed','multiply']:

			image_class = tf.keras.Input(shape=(1,), dtype='int32')
			class_embed = tf.keras.layers.Embedding(input_dim = self.num_classes, output_dim = 10, embeddings_initializer='glorot_normal')(image_class)
			class_embed = layers.Flatten()(class_embed)
			class_den1 = layers.Dense(1, use_bias=False,kernel_initializer=init_fn)(class_embed)
			inner_prod = tf.keras.layers.Multiply()([avg_pool, class_den1])

		real_vs_fake = layers.Add()([dense2, inner_prod])

		model = tf.keras.Model(inputs = [inputs,image_class], outputs= real_vs_fake)

		return model

	### NEED TO FIX WITH SELF VARS
	def show_result_cifar10(self, images = None, num_epoch = 0, show = False, save = False, path = 'result.png'):

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
		logger.setLevel(old_level)


	def CIFAR10_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(80,80,3), classes=1000)

	def FID_cifar10(self):

		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.resize(image,[80,80])
			return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)

			self.fid_train_images_names = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images_names)
			self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

			self.CIFAR10_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:

				noise, input_class = self.get_noise('test',self.fid_batch_size)
				if self.label_style == 'base':
					input_class = tf.one_hot(np.squeeze(input_class), depth = self.num_clsses)
				preds = self.generator([noise,input_class], training=False)				
				preds = tf.image.resize(preds, [80,80])
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
