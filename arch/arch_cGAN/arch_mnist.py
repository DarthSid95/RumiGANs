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
		# init_fn = tf.function(init_fn, autograph=False)
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
		# model.add(layers.BatchNormalization())
		# model.add(layers.ReLU(max_value = 1.))

		model = tf.keras.Model(inputs= [noise_ip, image_class], outputs = out)

		return model

	def discriminator_model_mnist(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		# init_fn = tf.function(init_fn, autograph=False)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape = (self.output_size,self.output_size,1))

		conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1])(inputs)
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
		# if self.gan == 'SGAN':
		# 	real_vs_fake = layers.Activation( activation = 'sigmoid')(real_vs_fake)

		model = tf.keras.Model(inputs = [inputs,image_class], outputs= real_vs_fake)

		return model


	### NEED TO FIX WITH SELF VARS
	def show_result_mnist(self, images=None, num_epoch=0, show = False, save = False, path = 'result.png'):
		images = tf.reshape(images, [images.shape[0],self.output_size,self.output_size,1])
		images_on_grid = self.image_grid(input_tensor = images, grid_shape = (self.num_to_print,self.num_to_print),image_shape=(self.output_size,self.output_size),num_channels=1)
		fig = plt.figure(figsize=(7,7))
		ax1 = fig.add_subplot(111)
		ax1.cla()
		ax1.axis("off")
		ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		label = 'Epoch {0}'.format(num_epoch)
		plt.title(label, fontsize=8)
		if save:
			plt.tight_layout()
			plt.savefig(path)
		if show:
			plt.show()
		else:
			plt.close()	


		# size_figure_grid = 10
		# fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(10, 10))
		# for i in range(size_figure_grid):
		# 	for j in range(size_figure_grid):
		# 		ax[i, j].get_xaxis().set_visible(False)
		# 		ax[i, j].get_yaxis().set_visible(False)

		# for k in range(size_figure_grid*size_figure_grid):
		# 	i = k // size_figure_grid
		# 	j = k % size_figure_grid
		# 	ax[i, j].cla()
		# 	ax[i, j].imshow(images[k,:,:,0], cmap='gray')

		# label = 'Epoch {0}'.format(num_epoch)
		# fig.text(0.5, 0.04, label, ha='center')

		# if save:
		# 	plt.savefig(path)

		# if show:
		# 	plt.show()
		# else:
		# 	plt.close()


	def MNIST_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(80,80,3), classes=1000)



	def FID_mnist(self):

		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.resize(image,[80,80])
				# This will convert to float values in [0, 1]

				# image = tf.divide(image,255.0)
				image = tf.image.grayscale_to_rgb(image)
				# image = tf.subtract(image,0.5)
				# image = tf.scalar_mul(2.0,image)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1
			self.FID_vec_even = []
			self.FID_vec_odd = []
			self.FID_vec_sharp = []
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

			random_points_sharp = tf.keras.backend.random_uniform([min(self.fid_images_sharp.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_images_sharp.shape[0]), dtype='int32', seed=None)
			self.fid_images_sharp = self.fid_images_sharp[random_points_sharp]
			self.fid_image_dataset_sharp = tf.data.Dataset.from_tensor_slices(self.fid_images_sharp)
			self.fid_image_dataset_sharp = self.fid_image_dataset_sharp.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset_sharp = self.fid_image_dataset_sharp.batch(self.fid_batch_size)

			random_points_single = tf.keras.backend.random_uniform([min(self.fid_images_single.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_images_single.shape[0]), dtype='int32', seed=None)
			self.fid_images_single = self.fid_images_single[random_points_single]
			self.fid_image_dataset_single = tf.data.Dataset.from_tensor_slices(self.fid_images_single)
			self.fid_image_dataset_single = self.fid_image_dataset_single.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset_single = self.fid_image_dataset_single.batch(self.fid_batch_size)

			# self.fid_train_images = tf.image.resize(self.fid_train_images, [80,80])
			# self.fid_train_images = tf.image.grayscale_to_rgb(self.fid_train_images)
			# self.fid_images = self.fid_train_images.numpy()

			self.MNIST_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			# print('logs/130919_ELeGANt_mnist_lsgan_base_01/130919_ELeGANt_mnist_lsgan_base_Results_checkpoints')
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			# for images_batch in self.fid_image_dataset_even:
			# 	input_class = np.expand_dims(np.random.choice([0,2,4,6,8], self.fid_batch_size), axis = 1).astype('int32')
			# 	noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
			# 	preds = self.generator([noise, input_class], training=False)
			# 	# preds = preds[:,:,:].numpy()		
			# 	preds = tf.image.resize(preds, [299,299])
			# 	preds = tf.image.grayscale_to_rgb(preds)
			# 	# preds = tf.subtract(preds,0.50)
			# 	# preds = tf.scalar_mul(2.0,preds)
			# 	preds = preds.numpy()

			# 	act1 = self.FID_model.predict(images_batch)
			# 	act2 = self.FID_model.predict(preds)
			# 	try:
			# 		self.act1 = np.concatenate([self.act1,act1], axis = 0)
			# 		self.act2 = np.concatenate([self.act2,act2], axis = 0)
			# 	except:
			# 		self.act1 = act1
			# 		self.act2 = act2
			# # print(self.act1.shape, self.act2.shape)
			# self.eval_FID()
			# self.FID_vec_even.append([self.fid, self.total_count.numpy()])

			# for images_batch in self.fid_image_dataset_odd:
			# 	input_class = np.expand_dims(np.random.choice([1,3,5,7,9], self.fid_batch_size), axis = 1).astype('int32')
			# 	noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
			# 	preds = self.generator([noise, input_class], training=False)
			# 	# preds = preds[:,:,:].numpy()		
			# 	preds = tf.image.resize(preds, [299,299])
			# 	preds = tf.image.grayscale_to_rgb(preds)
			# 	# preds = tf.subtract(preds,0.50)
			# 	# preds = tf.scalar_mul(2.0,preds)
			# 	preds = preds.numpy()

			# 	act1 = self.FID_model.predict(images_batch)
			# 	act2 = self.FID_model.predict(preds)
			# 	try:
			# 		self.act1 = np.concatenate([self.act1,act1], axis = 0)
			# 		self.act2 = np.concatenate([self.act2,act2], axis = 0)
			# 	except:
			# 		self.act1 = act1
			# 		self.act2 = act2
			# # print(self.act1.shape, self.act2.shape)
			# self.eval_FID()
			# self.FID_vec_odd.append([self.fid, self.total_count.numpy()])

			# for images_batch in self.fid_image_dataset_sharp:
			# 	input_class = np.expand_dims(np.random.choice([1,2,4,5,7,9],self.fid_batch_size), axis = 1).astype('int32')
			# 	noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
			# 	preds = self.generator([noise, input_class], training=False)
			# 	# preds = preds[:,:,:].numpy()		
			# 	preds = tf.image.resize(preds, [299,299])
			# 	preds = tf.image.grayscale_to_rgb(preds)
				# preds = tf.subtract(preds,0.50)
				# preds = tf.scalar_mul(2.0,preds)
			# 	preds = preds.numpy()

			# 	act1 = self.FID_model.predict(images_batch)
			# 	act2 = self.FID_model.predict(preds)
			# 	try:
			# 		self.act1 = np.concatenate([self.act1,act1], axis = 0)
			# 		self.act2 = np.concatenate([self.act2,act2], axis = 0)
			# 	except:
			# 		self.act1 = act1
			# 		self.act2 = act2
			# # print(self.act1.shape, self.act2.shape)
			# self.eval_FID()
			# self.FID_vec_sharp.append([self.fid, self.total_count.numpy()])

			for images_batch in self.fid_image_dataset_single:
				input_class = self.number*np.ones((self.fid_batch_size,1)).astype('int32')
				noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator([noise, input_class], training=False)
				# preds = preds[:,:,:].numpy()		
				preds = tf.image.resize(preds, [80,80])
				preds = tf.image.grayscale_to_rgb(preds)
				# preds = tf.subtract(preds,0.50)
				# preds = tf.scalar_mul(2.0,preds)
				preds = preds.numpy()

				act1 = self.FID_model.predict(images_batch)
				act2 = self.FID_model.predict(preds)
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			# print(self.act1.shape, self.act2.shape)
			self.eval_FID()
			self.FID_vec_single.append([self.fid, self.total_count.numpy()])

			return


	# def FID_mnist(self):
	# 	if self.FID_load_flag == 0:
	# 		### First time FID call setup
	# 		self.FID_load_flag = 1
	# 		self.FID_single_vec = []
	# 		self.FID_even_vec = []
	# 		self.FID_odd_vec = []
	# 		self.FID_sharp_vec = []
	# 		# if self.testcase == 'single':
	# 		# 	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
	# 		# 	train_images = train_images.reshape(train_images.shape[0], self.output_size,self.output_size,1).astype('float64')
	# 		# 	train_images = (train_images - 0.) / 255.0
	# 		# 	train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
	# 		# 	zero = train_labels == 0
	# 		# 	one = train_labels == 1
	# 		# 	two  = train_labels == 2
	# 		# 	three  = train_labels == 3
	# 		# 	four  = train_labels == 4
	# 		# 	five  = train_labels == 5
	# 		# 	six  = train_labels == 6
	# 		# 	seven  = train_labels == 7
	# 		# 	eight = train_labels == 8
	# 		# 	nine = train_labels == 9


	# 		# 	if self.testcase == 'single':	
	# 		# 		self.fid_train_images = train_images[np.where(train_labels == self.number)[0]] #train_images[np.where(train_labels == self.number)[0][0:500]]
	# 		# 	elif self.testcase == 'even':
	# 		# 		self.fid_train_images = train_images[np.where(train_labels%2 == 0)[0]]
	# 		# 	elif self.testcase == 'odd':
	# 		# 		self.fid_train_images = train_images[np.where(train_labels%2 != 0)[0]]
	# 		# 	elif self.testcase == 'sharp':
	# 		# 		self.fid_train_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
	# 		# else:
	# 		# 	self.fid_train_images = train_images
	# 		# else:
	# 		# 	self.fid_train_images = self.train_data
	# 		random_points = tf.keras.backend.random_uniform([1000], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
	# 		print(random_points)
	# 		self.fid_train_images = self.fid_train_images[random_points]
	# 		self.fid_train_images = tf.image.resize(self.fid_train_images, [80,80])
	# 		self.fid_train_images = tf.image.grayscale_to_rgb(self.fid_train_images)
	# 		self.fid_train_images = self.fid_train_images.numpy()

	# 		self.MNIST_Classifier()


	# 	if self.mode == 'fid':
	# 		print(self.checkpoint_dir)
	# 		# print('logs/130919_ELeGANt_mnist_lsgan_base_01/130919_ELeGANt_mnist_lsgan_base_Results_checkpoints')
	# 		self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
	# 		print('Models Loaded Successfully')

	# 	###SINGLEE
	# 	# print(self.fid_train_images.shape)
	# 	# if self.testcase == 'single':
	# 		# input_class = (np.ones([self.fid_train_images.shape[0],1])*self.number).astype('int32')
	# 	# elif self.testcase =='even':
	# 	# 	input_class = np.expand_dims(np.random.choice([0,2,4,6,8], self.fid_train_images.shape[0]), axis = 1).astype('int32')
	# 	# elif self.testcase =='odd':
	# 	# 	input_class = np.expand_dims(np.random.choice([1,3,5,7,9], self.fid_train_images.shape[0]), axis = 1).astype('int32')
	# 	# elif self.testcase =='sharp':
	# 	# 	input_class = np.expand_dims(np.random.choice([1,2,4,5,7,9], self.fid_train_images.shape[0]), axis = 1).astype('int32')
		
	# 	input_class = (np.ones([self.fid_train_images.shape[0],1])*self.number).astype('int32')
	# 	preds = self.generator([tf.random.normal([self.fid_train_images.shape[0], self.noise_dims]), input_class], training=False)
	# 	# preds = preds[:,:,:].numpy()		
	# 	preds = tf.image.resize(preds, [80,80])
	# 	preds = tf.image.grayscale_to_rgb(preds)
	# 	preds = preds.numpy()

	# 	# calculate latent representations
	# 	self.act1 = self.FID_model.predict(self.fid_train_images)
	# 	self.act2 = self.FID_model.predict(preds)
	# 	self.eval_FID()
	# 	self.FID_single_vec.append(self.fid)
	# 	path = self.impath
	# 	np.save(path+'_single_FID.npy',np.array(self.FID_single_vec))

	# 	#EVENN
	# 	input_class = np.expand_dims(np.random.choice([0,2,4,6,8], self.fid_train_images.shape[0]), axis = 1).astype('int32')

	# 	preds = self.generator([tf.random.normal([self.fid_train_images.shape[0], self.noise_dims]), input_class], training=False)
	# 	# preds = preds[:,:,:].numpy()		
	# 	preds = tf.image.resize(preds, [80,80])
	# 	preds = tf.image.grayscale_to_rgb(preds)
	# 	preds = preds.numpy()

	# 	# calculate latent representations
	# 	self.act1 = self.FID_model.predict(self.fid_train_images)
	# 	self.act2 = self.FID_model.predict(preds)
	# 	self.eval_FID()
	# 	self.FID_even_vec.append(self.fid)
	# 	path = self.impath
	# 	np.save(path+'_even_FID.npy',np.array(self.FID_even_vec))

	# 	####ODDD
	# 	input_class = np.expand_dims(np.random.choice([1,3,5,7,9], self.fid_train_images.shape[0]), axis = 1).astype('int32')

	# 	preds = self.generator([tf.random.normal([self.fid_train_images.shape[0], self.noise_dims]), input_class], training=False)
	# 	# preds = preds[:,:,:].numpy()		
	# 	preds = tf.image.resize(preds, [80,80])
	# 	preds = tf.image.grayscale_to_rgb(preds)
	# 	preds = preds.numpy()

	# 	# calculate latent representations
	# 	self.act1 = self.FID_model.predict(self.fid_train_images)
	# 	self.act2 = self.FID_model.predict(preds)
	# 	self.eval_FID()
	# 	self.FID_odd_vec.append(self.fid)
	# 	path = self.impath
	# 	np.save(path+'_odd_FID.npy',np.array(self.FID_odd_vec))

	# 	### SHARPPPP
	# 	input_class = np.expand_dims(np.random.choice([1,2,4,5,7,9], self.fid_train_images.shape[0]), axis = 1).astype('int32')

	# 	preds = self.generator([tf.random.normal([self.fid_train_images.shape[0], self.noise_dims]), input_class], training=False)
	# 	# preds = preds[:,:,:].numpy()		
	# 	preds = tf.image.resize(preds, [80,80])
	# 	preds = tf.image.grayscale_to_rgb(preds)
	# 	preds = preds.numpy()

	# 	# calculate latent representations
	# 	self.act1 = self.FID_model.predict(self.fid_train_images)
	# 	self.act2 = self.FID_model.predict(preds)
	# 	self.eval_FID()
	# 	self.FID_sharp_vec.append(self.fid)
	# 	path = self.impath
	# 	np.save(path+'_sharp_FID.npy',np.array(self.FID_sharp_vec))




	# def MNIST_Classifier(self):
	# 	self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(80,80,3), classes=1000)

	# def FID_mnist(self):
	# 	self.MNIST_Classifier()
	# 	if self.mode == 'fid':
	# 		print(self.checkpoint_dir)
	# 		# print('logs/130919_ELeGANt_mnist_lsgan_base_01/130919_ELeGANt_mnist_lsgan_base_Results_checkpoints')
	# 		self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
	# 		print('Models Loaded Successfully')
	# 	else:
	# 		print('Evaluating FID Score ')

	# 	with tf.device('/CPU'):
	# 		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
	# 		train_images = train_images.reshape(train_images.shape[0], self.output_size,self.output_size,1).astype('float64')
	# 		train_images = (train_images - 0.) / 255.0

	# 		if self.testcase == 'single':
	# 			t_images = train_images[np.where(train_labels == self.number)[0][0:5000]]
	# 			train_images = t_images
	# 		if self.testcase == 'even':
	# 			train_images = train_images[np.where(train_labels%2 == 0)[0]]
	# 		if self.testcase == 'odd':
	# 			train_images = train_images[np.where(train_labels%2 != 0)[0]]

	# 			# for i in range(self.reps-1):
	# 			# 	train_images = np.concatenate([train_images, t_images])

	# 		noise = tf.random.normal([train_images.shape[0], self.noise_dims])
	# 		preds = self.generator(noise, training=False)
	# 		preds = preds[:,:,:,0].numpy()
	# 		# calculate latent representations
	# 		self.act1 = self.FID_model.predict(train_images)
	# 		self.act2 = self.FID_model.predict(np.expand_dims(preds,axis=3))

	# 		# prd_data_1 = prd.compute_prd_from_embedding(self.act2, self.act1)


	# 		# self.tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
	# 		# plot_only = 500
	# 		# low_dim_embs = tsne.fit_transform(self.act1)
	# 		# for i in range(10):
	# 		# 		x, y = low_dim_embs[i*6000:((i+1)*6000), :]
	# 		# 		print(x,x.shape)
	# 		# 		print(y,y.shape)
	# 		# 		plt.scatter(x, y)
	# 		# 		# plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
	# 		# plt.savefig(self.tsne_path)


	# 		#### COMMENTED FOR GPU0 ISSUE 
	# 		self.eval_FID()
	# 		if self.testcase != 'single':
	# 			print("FID SCORE: {:>2.4f}\n".format(self.fid))
	# 			if self.res_flag:
	# 				self.res_file.write("FID SCORE: {:>2.4f}\n".format(self.fid))
	# 		else:
	# 			print("FID SCORE Full: {:>2.4f}\n".format(self.fid))
	# 			if self.res_flag:
	# 				self.res_file.write("FID SCORE Full: {:>2.4f}\n".format(self.fid))

	# 			self.reps = 500
	# 			t_images = train_images[np.where(train_labels == self.number)[0][0:10]]
	# 			train_images = t_images
	# 			for i in range(self.reps-1):
	# 				train_images = np.concatenate([train_images, t_images])
	# 			self.act1 = self.FID_model.predict(train_images)
	# 			self.eval_FID()
	# 			print("FID SCORE Small: {:>2.4f}\n".format(self.fid))
	# 			if self.res_flag:
	# 				self.res_file.write("FID SCORE Small: {:>2.4f}\n".format(self.fid))

	# 		# prd_data_2 = prd.compute_prd_from_embedding(self.act2, self.act1)
	# 		# prd.plot([prd_data_1, prd_data_2], ['GAN_1', 'GAN_2'], out_path = self.tsne_path)

	# def MNIST_Classifier(self):

	# 	if os.path.exists('MNIST_FID.h5'):
	# 		print('Existing MNIST encoder model is being loaded')
	# 		# self.FID_model = tf.keras.models.load_model('MNIST_FID.h5')
	# 		with tf.device(self.device):
	# 			if not self.FID_load_flag:
	# 				self.FID_model = tf.keras.models.load_model('MNIST_FID.h5')
	# 				self.FID_load_flag = 1
	# 		# self.FID_model.layers.pop()
	# 		# self.FID_model.layers.pop()

	# 		# model = tf.keras.Sequential()
	# 		# model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
	# 		#                  activation='relu',
	# 		#                  input_shape=(28,28,1)))
	# 		# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	# 		# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
	# 		# model.add(tf.keras.layers.Dropout(0.25))
	# 		# model.add(tf.keras.layers.Flatten())
	# 		# model.add(tf.keras.layers.Dense(128, activation='relu'))

	# 		# model.set_weights(self.FID_model.get_weights()[0:6])

	# 		# model.summary()
	# 		# model.save('MNIST_FID.h5')
	# 		# self.FID_model.build()
	# 		print(self.FID_model.summary())
	# 		return

	# 	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
	# 	train_images = train_images.reshape(train_images.shape[0], self.output_size,self.output_size,1).astype('float64')
	# 	train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float64')
	# 	train_images = (train_images  -127.5) / 255.0

	# 	train_cat = tf.keras.utils.to_categorical(train_labels,num_classes=10,dtype='float64')


	# 	init_fn = tf.keras.initializers.glorot_uniform()
	# 	init_fn = tf.function(init_fn, autograph=False)


	# 	''' MNIST Classifier for FID'''
	# 	inputs = tf.keras.Input(shape=(self.output_size,self.output_size,1))

	# 	x1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu')(inputs)
	# 	x2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x1)
	# 	x3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)
	# 	x4 = tf.keras.layers.Flatten()(x3)
	# 	x5 = tf.keras.layers.Dense(128, activation='relu')(x4)
	# 	x6 = tf.keras.layers.Dense(100, activation='relu')(x5)
	# 	x7 = tf.keras.layers.Dense(50, activation='relu')(x6)
	# 	Cla = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x7)



	# 	FID = tf.keras.Model(inputs=inputs, outputs=x7)
	# 	Classifier = tf.keras.Model(inputs=inputs, outputs=Cla)

	# 	''' Autoencoder Architecture for FID'''
	# 	# inputs = tf.keras.Input(shape=(self.output_size,self.output_size,))
	# 	# reshape = tf.keras.layers.Reshape(target_shape=(self.output_size*self.output_size,))(inputs)
	# 	# x1 = tf.keras.layers.Dense(50, activation=tf.nn.relu)(reshape)
	# 	# # x2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x1)
	# 	# Enc = tf.keras.layers.Dense(8, activation=tf.nn.relu)(x1)
	# 	# Encoder = tf.keras.Model(inputs=inputs, outputs=Enc)
	# 	# # y1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(Enc)
	# 	# # y2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(y1)
	# 	# y1 = tf.keras.layers.Dense(50, activation=tf.nn.relu)(Enc)
	# 	# Decoder = tf.keras.layers.Dense(784, activation=tf.nn.relu)(y1)
	# 	# Decoder = tf.keras.layers.Reshape(target_shape=(self.output_size,self.output_size,))(Decoder)
	# 	# Autoencoder = tf.keras.Model(inputs=inputs, outputs=Decoder)

	# 	print('FID training model made')
	# 	print(FID.summary(),Classifier.summary())

	# 	Classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# 	Classifier.fit(train_images, train_cat,epochs=10,batch_size=100,shuffle=True,)
	# 	Classifier.save("MNIST_FID_FULL.h5")
	# 	FID.save("MNIST_FID.h5")

	# 	self.MNIST_Classifier()
	# 	return

