from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from ops import *

class ARCH_mnist():

	def __init__(self):
		print("Creating MNIST architectures for cGAN cases ")
		return

	def generator_model_mnist(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		
		if self.label_style == 'base':

			noise_ip = tf.keras.Input(shape=(self.noise_dims, ))
			image_class = tf.keras.Input(shape=(self.num_classes,))
			gen_concat = tf.keras.layers.Concatenate()([noise_ip, image_class])
			gen_dense = layers.Dense(int(self.output_size/4)*int(self.output_size/4)*64)(gen_concat)
			gen_ip = layers.Reshape((int(self.output_size/4), int(self.output_size/4), 64))(gen_dense)

		elif self.label_style == 'embed':

			noise_ip = tf.keras.Input(shape=(self.noise_dims, ))
			image_class = tf.keras.Input(shape=(1,), dtype='int32')

			noise_den = layers.Dense(int(self.output_size/4)*int(self.output_size/4)*63, use_bias=False,kernel_initializer=init_fn)(noise_ip)
			noise_res =layers.Reshape((int(self.output_size/4), int(self.output_size/4),63))(noise_den)

			class_embed = tf.keras.layers.Embedding(input_dim = self.num_classes, output_dim = 10, embeddings_initializer='glorot_normal')(image_class)
			class_den = layers.Dense(int(self.output_size/4)*int(self.output_size/4), use_bias=False,kernel_initializer=init_fn)(class_embed)
			class_res = layers.Reshape((int(self.output_size/4), int(self.output_size/4), 1))(class_den)
			gen_ip = tf.keras.layers.Concatenate()([noise_res, class_res])

		elif self.label_style == 'multiply':

			noise_ip = tf.keras.Input(shape=(self.noise_dims, ))
			image_class = tf.keras.Input(shape=(1,), dtype='int32')

			class_embed = tf.keras.layers.Embedding(input_dim = self.num_classes, output_dim = self.noise_dims, embeddings_initializer='glorot_normal')(image_class)
			
			gen_multiply = tf.keras.layers.Multiply()([noise_ip,class_embed])
			gen_dense = layers.Dense(int(self.output_size/4)*int(self.output_size/4)*64)(gen_multiply)
			gen_ip = layers.Reshape((int(self.output_size/4), int(self.output_size/4), 64))(gen_dense)


		deconv1 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn)(gen_ip)
		deconv1 = layers.BatchNormalization()(deconv1)
		deconv1 = layers.LeakyReLU()(deconv1)

		deconv2 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn)(deconv1)
		deconv2 = layers.BatchNormalization()(deconv2)
		deconv2 = layers.LeakyReLU()(deconv2)

		deconv3 = layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn)(deconv2)
		deconv3 = layers.BatchNormalization()(deconv3)
		deconv3 = layers.LeakyReLU()(deconv3)

		deconv4 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn,)(deconv3)
		deconv4 = layers.BatchNormalization()(deconv4)
		deconv4 = layers.LeakyReLU()(deconv4)

		out = layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn)(deconv4)
		out = layers.Activation( activation = 'tanh')(out)

		model = tf.keras.Model(inputs= [noise_ip, image_class], outputs = out)

		return model

	def discriminator_model_mnist(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape = (self.output_size,self.output_size,1))

		conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1])(inputs)
		conv1 = layers.BatchNormalization()(conv1)
		conv1 = layers.LeakyReLU()(conv1)

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


	def MNIST_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(80,80,3), classes=1000)



	def FID_mnist(self):

		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.resize(image,[80,80])
				image = tf.image.grayscale_to_rgb(image)
			return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1
			self.FID_vec_even = []
			self.FID_vec_odd = []
			self.FID_vec_overlap = []
			self.FID_vec_single = []

			random_points_even = tf.keras.backend.random_uniform([min(self.fid_images_even.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_images_even.shape[0]), dtype='int32', seed=None)
			self.fid_images_even = self.fid_images_even[random_points_even]
			self.fid_image_dataset_even = tf.data.Dataset.from_tensor_slices(self.fid_images_even)
			self.fid_image_dataset_even = self.fid_image_dataset_even.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset_even = self.fid_image_dataset_even.batch(self.fid_batch_size)

			### Added for NIPS Rebuttal
			random_points_odd = tf.keras.backend.random_uniform([min(self.fid_images_odd.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_images_odd.shape[0]), dtype='int32', seed=None)
			self.fid_images_odd = self.fid_images_odd[random_points_odd]
			self.fid_image_dataset_odd = tf.data.Dataset.from_tensor_slices(self.fid_images_odd)
			self.fid_image_dataset_odd = self.fid_image_dataset_odd.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset_odd = self.fid_image_dataset_odd.batch(self.fid_batch_size)

			random_points_overlap = tf.keras.backend.random_uniform([min(self.fid_images_overlap.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_images_overlap.shape[0]), dtype='int32', seed=None)
			self.fid_images_overlap = self.fid_images_overlap[random_points_overlap]
			self.fid_image_dataset_overlap = tf.data.Dataset.from_tensor_slices(self.fid_images_overlap)
			self.fid_image_dataset_overlap = self.fid_image_dataset_overlap.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset_overlap = self.fid_image_dataset_overlap.batch(self.fid_batch_size)

			random_points_single = tf.keras.backend.random_uniform([min(self.fid_images_single.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_images_single.shape[0]), dtype='int32', seed=None)
			self.fid_images_single = self.fid_images_single[random_points_single]
			self.fid_image_dataset_single = tf.data.Dataset.from_tensor_slices(self.fid_images_single)
			self.fid_image_dataset_single = self.fid_image_dataset_single.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset_single = self.fid_image_dataset_single.batch(self.fid_batch_size)

			self.MNIST_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for images_batch in self.fid_image_dataset_even:
				input_class = np.expand_dims(np.random.choice([0,2,4,6,8], self.fid_batch_size), axis = 1).astype('int32')
				noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator([noise, input_class], training=False)
				preds = tf.image.resize(preds, [299,299])
				preds = tf.image.grayscale_to_rgb(preds)
				preds = preds.numpy()

				act1 = self.FID_model.predict(images_batch)
				act2 = self.FID_model.predict(preds)
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			self.eval_FID()
			self.FID_vec_even.append([self.fid, self.total_count.numpy()])

			for images_batch in self.fid_image_dataset_odd:
				input_class = np.expand_dims(np.random.choice([1,3,5,7,9], self.fid_batch_size), axis = 1).astype('int32')
				noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator([noise, input_class], training=False)
				preds = tf.image.resize(preds, [299,299])
				preds = tf.image.grayscale_to_rgb(preds)
				preds = preds.numpy()

				act1 = self.FID_model.predict(images_batch)
				act2 = self.FID_model.predict(preds)
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			self.eval_FID()
			self.FID_vec_odd.append([self.fid, self.total_count.numpy()])

			for images_batch in self.fid_image_dataset_overlap:
				input_class = np.expand_dims(np.random.choice([1,2,4,5,7,9],self.fid_batch_size), axis = 1).astype('int32')
				noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator([noise, input_class], training=False)
						
				preds = tf.image.resize(preds, [299,299])
				preds = tf.image.grayscale_to_rgb(preds)
				preds = tf.subtract(preds,0.50)
				preds = tf.scalar_mul(2.0,preds)
				preds = preds.numpy()

				act1 = self.FID_model.predict(images_batch)
				act2 = self.FID_model.predict(preds)
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			self.eval_FID()
			self.FID_vec_overlap.append([self.fid, self.total_count.numpy()])

			for images_batch in self.fid_image_dataset_single:
				input_class = self.number*np.ones((self.fid_batch_size,1)).astype('int32')
				noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator([noise, input_class], training=False)
				preds = tf.image.resize(preds, [80,80])
				preds = tf.image.grayscale_to_rgb(preds)
				preds = preds.numpy()

				act1 = self.FID_model.predict(images_batch)
				act2 = self.FID_model.predict(preds)
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			self.eval_FID()
			self.FID_vec_single.append([self.fid, self.total_count.numpy()])

			return

