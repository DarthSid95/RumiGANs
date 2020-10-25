from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class ARCH_mnist():

	def __init__(self):
		print("Creating MNIST architectures for SubGAN cases ")
		return

	def generator_model_mnist(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(int(self.output_size/4)*int(self.output_size/4)*256, use_bias=False, input_shape=(self.noise_dims,),kernel_initializer=init_fn))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Reshape((int(self.output_size/4), int(self.output_size/4), 256)))
		assert model.output_shape == (None, int(self.output_size/4), int(self.output_size/4), 256) # Note: None is the batch size

		model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/4), int(self.output_size/4), 128)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/2), int(self.output_size/2), 64)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size/2), int(self.output_size/2), 32)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn,))
		assert model.output_shape == (None, int(self.output_size), int(self.output_size), 1)
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn))
		assert model.output_shape == (None, int(self.output_size), int(self.output_size), 1)
		model.add(layers.Activation( activation = 'tanh'))

		return model

	def discriminator_model_mnist(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn))
		model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())
		
		model.add(layers.Flatten())
		
		model.add(layers.Dense(50))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))

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
			self.FID_vec_pos = []
			self.FID_vec_neg = []
			random_points = tf.keras.backend.random_uniform([min(self.fid_train_images.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images = self.fid_train_images[random_points]
			self.fid_image_dataset_pos = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			self.fid_image_dataset_pos = self.fid_image_dataset_pos.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset_pos = self.fid_image_dataset_pos.batch(self.fid_batch_size)

			### Added for NIPS Rebuttal
			random_points_neg = tf.keras.backend.random_uniform([min(self.fid_others.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_others.shape[0]), dtype='int32', seed=None)
			print(random_points_neg)
			self.fid_others = self.fid_others[random_points_neg]
			self.fid_image_dataset_neg = tf.data.Dataset.from_tensor_slices(self.fid_others)
			self.fid_image_dataset_neg = self.fid_image_dataset_neg.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset_neg = self.fid_image_dataset_neg.batch(self.fid_batch_size)


			self.MNIST_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for image_batch,image_batch_other in zip(self.fid_image_dataset_pos,self.fid_image_dataset_neg):
				noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator(noise, training=False)
				preds = tf.image.resize(preds, [80,80])
				preds = tf.image.grayscale_to_rgb(preds)
				preds = preds.numpy()

				act1 = self.FID_model.predict(image_batch)
				act2 = self.FID_model.predict(preds)
				act3 = self.FID_model.predict(image_batch_other)
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
					self.act3 = np.concatenate([self.act3,act3], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
					self.act3 = act3
			# print(self.act1.shape, self.act2.shape)
			self.eval_FID()
			self.FID_vec_pos.append([self.fid, self.total_count.numpy()])

			### Added for NIPS Rebuttal
			self.act1_temp = self.act1
			self.act1 = self.act3
			self.eval_FID()
			self.FID_vec_neg.append([self.fid, self.total_count.numpy()])

			### Need to fix "Added for NIPS Rebuttal" 
			self.act1 = self.act1_temp
			return

	def MNIST_VGG16_Classifier(self):
		self.classifier_model = tf.keras.models.load_model('data/MNIST/MNIST_Classifier_Models/VGG16.h5')

	def class_prob_metric(self):
		
		if self.classifier_load_flag == 0:
			### First time FID call setup
			self.classifier_load_flag = 1
			self.MNIST_VGG16_Classifier()

		noise = tf.random.normal([6000, self.noise_dims],self.noise_mean, self.noise_stddev)
		preds = self.generator(noise, training=False)
		classifics  = self.classifier_model(preds)
		classifics_hist = tf.reduce_mean(classifics, axis = 0).numpy()
		print("Class-Wise Average Probabilities when alpha_p = " +str(self.alphap)+ "\n")
		print(classifics_hist)
		if self.res_flag:
			self.res_file.write("Class-Wise Average Probabilities alpha_p = "+str(self.alphap)+ "\n")
			self.res_file.write(str(classifics_hist))
			self.res_file.write("\n")

		self.class_prob_vec.append([classifics_hist, self.total_count.numpy()])
