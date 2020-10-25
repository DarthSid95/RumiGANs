from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class ARCH_cifar10():
	def __init__(self):
		print("Creating CIFAR-10 architectures for SubGAN case")
		return

	def generator_model_cifar10(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape = (self.noise_dims,))

		enc_res = tf.keras.layers.Reshape([1,1,int(self.noise_dims)])(inputs) #1x1xlatent

		denc4 = tf.keras.layers.Conv2DTranspose(512, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(dense) #2x2x128
		denc4 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc4)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc4)
		denc4 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc4)

		denc3 = tf.keras.layers.Conv2DTranspose(256, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(denc4) #4x4x256
		denc3 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc3)
		# denc3 = tf.keras.layers.Dropout(0.5)(denc3)
		denc3 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc3)


		denc2 = tf.keras.layers.Conv2DTranspose(128, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(denc3) #8x8x128
		denc2 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc2)
		# denc2 = tf.keras.layers.Dropout(0.5)(denc2)
		denc2 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc2)


		denc1 = tf.keras.layers.Conv2DTranspose(64, 5, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(denc2) #16x16x64
		denc1 = tf.keras.layers.BatchNormalization(momentum=0.9)(denc1)
		# denc1 = tf.keras.layers.Dropout(0.5)(denc1)
		denc1 = tf.keras.layers.LeakyReLU(alpha=0.1)(denc1)

		out = tf.keras.layers.Conv2DTranspose(3, 5,strides=2,padding='same', kernel_initializer=init_fn)(denc1) #32x32x3
		out =  tf.keras.layers.Activation( activation = 'tanh')(out)

		
		model = tf.keras.Model(inputs=inputs, outputs=out)
		return model

	def discriminator_model_cifar10(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential() #64x64x3
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn,input_shape=[self.output_size, self.output_size, 3])) #32x32x64
		model.add(layers.BatchNormalization(momentum=0.9))
		model.add(layers.LeakyReLU(alpha=0.1))

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #16x16x128
		model.add(layers.BatchNormalization(momentum=0.9))
		model.add(layers.LeakyReLU(alpha=0.1))

		model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #8x8x256
		model.add(layers.BatchNormalization(momentum=0.9))
		model.add(layers.LeakyReLU(alpha=0.1))

		model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn)) #4x4x512
		model.add(layers.BatchNormalization(momentum=0.9))
		model.add(layers.LeakyReLU(alpha=0.1))

		model.add(layers.Flatten()) #8192x1
		model.add(layers.Dense(512))	
		model.add(layers.Dense(1)) #1x1


		return model


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
			random_points = tf.keras.backend.random_uniform([1000], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
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
				noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator(noise, training=False)
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

