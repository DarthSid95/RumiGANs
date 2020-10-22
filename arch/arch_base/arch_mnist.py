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
		print("Creating MNIST architectures for base cases ")
		return

	def generator_model_mnist(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		# init_fn = tf.function(init_fn, autograph=False)
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
		# model.add(layers.BatchNormalization())
		# model.add(layers.ReLU(max_value = 1.))

		return model
		# init_fn = tf.keras.initializers.glorot_uniform()
		# init_fn = tf.function(init_fn, autograph=False)

		# model = tf.keras.Sequential()
		# model.add(layers.Dense(int(self.output_size/4)*int(self.output_size/4)*256, use_bias=False, input_shape=(self.noise_dims,),kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())

		# model.add(layers.Reshape((int(self.output_size/4), int(self.output_size/4), 256)))
		# assert model.output_shape == (None, int(self.output_size/4), int(self.output_size/4), 256) # Note: None is the batch size

		# model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn))
		# assert model.output_shape == (None, int(self.output_size/4), int(self.output_size/4), 128)
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())

		# model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn))
		# assert model.output_shape == (None, int(self.output_size/2), int(self.output_size/2), 64)
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())

		# model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn))
		# assert model.output_shape == (None, int(self.output_size/2), int(self.output_size/2), 32)
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())

		# model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=init_fn,))
		# assert model.output_shape == (None, int(self.output_size), int(self.output_size), 1)
		# model.add(layers.BatchNormalization())
		# model.add(layers.LeakyReLU())

		# model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=init_fn))
		# assert model.output_shape == (None, int(self.output_size), int(self.output_size), 1)
		# # model.add(layers.BatchNormalization())
		# # model.add(layers.ReLU(max_value = 1.))

		# return model

	#### Enc-Dec version added for ICML rebuttal. Uncomment only if needed
	# def generator_model_mnist(self):

	# 	init_fn = tf.keras.initializers.glorot_uniform()
	# 	init_fn = tf.function(init_fn, autograph=False)

	# 	inputs = tf.keras.Input(shape=(self.noise_dims,))

	# 	enc1 = tf.keras.layers.Dense(int(self.noise_dims), kernel_initializer=init_fn)(inputs)
	# 	enc1 = tf.keras.layers.LeakyReLU()(enc1)

	# 	enc2 = tf.keras.layers.Dense(128, kernel_initializer=init_fn)(enc1)
	# 	enc2 = tf.keras.layers.LeakyReLU()(enc2)

	# 	enc21 = tf.keras.layers.Dense(64, kernel_initializer=init_fn)(enc2)
	# 	enc21 = tf.keras.layers.LeakyReLU()(enc21)

	# 	enc22 = tf.keras.layers.Dense(64, kernel_initializer=init_fn)(enc21)
	# 	# enc22 = tf.keras.layers.LeakyReLU()(enc22)

	# 	enc3 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias=True)(enc22)
	# 	# enc3 = tf.keras.layers.LeakyReLU()(enc3)

	# 	enc4 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias=True)(enc3)
	# 	enc4 = tf.keras.layers.Activation( activation = 'sigmoid')(enc4)
	# 	# enc4 = tf.keras.layers.ReLU(max_value = 1.)(enc4)

	# 	model = tf.keras.Model(inputs = inputs, outputs = enc4)
		
	# 	return model


	#### Enc-Dec version added for ICML rebuttal. Uncomment only if needed
	def EncDec_model_mnist(self): 
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		# init_fn = tf.function(init_fn, autograph=False)

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,self.output_size,1))

		inputs_res = tf.keras.layers.Reshape([int(self.output_size*self.output_size),])(inputs)

		enc0 = tf.keras.layers.Dense(128, kernel_initializer=init_fn)(inputs_res)
		enc0 = tf.keras.layers.BatchNormalization()(enc0)
		# enc0 = tf.keras.layers.Dropout(0.3)(enc0)
		enc0 = tf.keras.layers.LeakyReLU()(enc0)

		enc1 = tf.keras.layers.Dense(64, kernel_initializer=init_fn)(enc0)
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		# enc1 = tf.keras.layers.Dropout(0.3)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(32, kernel_initializer=init_fn)(enc1)
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		# enc2 = tf.keras.layers.Dropout(0.3)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(32, kernel_initializer=init_fn)(enc2)
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		# enc3 = tf.keras.layers.Dropout(0.3)(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias=True)(enc3)
		enc4 = tf.keras.layers.Activation( activation = 'sigmoid')(enc4)
		# enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.ReLU(max_value = 1.)(enc4)
		### 11022020 - 05 onwards, tanh. sigmoid before that.

		encoded = tf.keras.Input(shape=(self.latent_dims,))

		dec0 = tf.keras.layers.Dense(32, kernel_initializer=init_fn)(encoded)
		dec0 = tf.keras.layers.BatchNormalization()(dec0)
		# dec0 = tf.keras.layers.Dropout(0.3)(dec0)
		dec0 = tf.keras.layers.LeakyReLU()(dec0)

		dec1 = tf.keras.layers.Dense(32, kernel_initializer=init_fn)(dec0)
		dec1 = tf.keras.layers.BatchNormalization()(dec1)
		# dec1 = tf.keras.layers.Dropout(0.3)(dec1)
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		dec2 = tf.keras.layers.Dense(64, kernel_initializer=init_fn)(dec1)
		dec2 = tf.keras.layers.BatchNormalization()(dec2)
		# dec2 = tf.keras.layers.Dropout(0.3)(dec2)
		dec2 = tf.keras.layers.LeakyReLU()(dec2)

		dec3 = tf.keras.layers.Dense(128, kernel_initializer=init_fn)(dec2)
		dec3 = tf.keras.layers.BatchNormalization()(dec3)
		# dec3 = tf.keras.layers.Dropout(0.3)(dec3)
		dec3 = tf.keras.layers.LeakyReLU()(dec3)

		out_enc = tf.keras.layers.Dense(int(self.output_size*self.output_size), kernel_initializer=init_fn)(dec3)
		out_enc = tf.keras.layers.Activation( activation = 'sigmoid')(out_enc)


		out = tf.keras.layers.Reshape([int(self.output_size),int(self.output_size),1])(out_enc)
		# out = tf.keras.layers.ReLU(max_value = 1.)(out)

		self.Encoder = tf.keras.Model(inputs = inputs, outputs = enc4)
		self.Decoder = tf.keras.Model(inputs = encoded, outputs = out)
		
		return self.Encoder

	def discriminator_model_mnist(self):
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
		# init_fn = tf.function(init_fn, autograph=False)
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())
		# model.add(layers.Dropout(0.3))

		model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())

		# model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		# model.add(layers.LeakyReLU())

		model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn))
		model.add(layers.LeakyReLU())
		model.add(layers.BatchNormalization())
		# model.add(layers.Dropout(0.3))

		model.add(layers.Flatten())
		
		model.add(layers.Dense(50))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# if self.gan == 'SGAN':
		# 	model.add(layers.Activation( activation = 'sigmoid'))
		# if self.gan_name =='SGAN':
		# 	model.add(tf.keras.layers.Activation( activation = 'sigmoid'))

		return model

	#### Enc-Dec version added for ICML rebuttal. Uncomment only if needed
	# def discriminator_model_mnist(self):
	# 	init_fn = tf.keras.initializers.glorot_uniform()
	# 	init_fn = tf.function(init_fn, autograph=False)

	# 	model = tf.keras.Sequential()
	# 	model.add(layers.Dense(256, use_bias=False, input_shape=(self.latent_dims,), kernel_initializer=init_fn))
	# 	# model.add(layers.BatchNormalization())
	# 	model.add(layers.LeakyReLU())

	# 	model.add(layers.Dense(512, use_bias=False, kernel_initializer=init_fn))
	# 	# model.add(layers.BatchNormalization())
	# 	model.add(layers.LeakyReLU())

	# 	model.add(layers.Dense(128, use_bias=False, kernel_initializer=init_fn))
	# 	# model.add(layers.BatchNormalization())
	# 	model.add(layers.LeakyReLU())

	# 	model.add(layers.Dense(1))
	# 	# model.add(layers.Softmax())
	# 	return model



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
		# size_figure_grid = 5
		# fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
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
			random_points = tf.keras.backend.random_uniform([min(self.fid_train_images.shape[0],self.FID_num_samples)], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images = self.fid_train_images[random_points]
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

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
			for image_batch in self.fid_image_dataset:
				# print(self.fid_train_images.shape)
				noise = tf.random.normal([self.fid_batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
				preds = self.generator(noise, training=False)
				# preds = preds[:,:,:].numpy()		
				preds = tf.image.resize(preds, [80,80])
				preds = tf.image.grayscale_to_rgb(preds)
				# preds = tf.subtract(preds,0.50)
				# preds = tf.scalar_mul(2.0,preds)
				preds = preds.numpy()

				act1 = self.FID_model.predict(image_batch)
				act2 = self.FID_model.predict(preds)
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			# print(self.act1.shape, self.act2.shape)
			self.eval_FID()
			return

	# def FID_mnist(self):
	# 	if self.FID_load_flag == 0:
	# 		### First time FID call setup
	# 		self.FID_load_flag = 1
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
	# 		# #elif self.testcase == 'even':
	# 		# #	self.fid_train_images = train_images[np.where(train_labels%2 == 0)[0]]
	# 		# #elif self.testcase == 'odd':
	# 		# #	self.fid_train_images = train_images[np.where(train_labels%2 != 0)[0]]
	# 		# #elif self.testcase == 'sharp':
	# 		# #	self.fid_train_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
	# 		# #else:
	# 		# #	self.fid_train_images = train_images
	# 		# else:
	# 		# 	self.fid_train_images = self.train_data
	# 		random_points = tf.keras.backend.random_uniform([5000], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
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

	# 	with tf.device('/CPU'):
	# 		# print(self.fid_train_images.shape)
	# 		preds = self.generator(tf.random.normal([self.fid_train_images.shape[0], self.noise_dims]), training=False)
	# 		# preds = preds[:,:,:].numpy()		
	# 		preds = tf.image.resize(preds, [80,80])
	# 		preds = tf.image.grayscale_to_rgb(preds)
	# 		preds = preds.numpy()

	# 		# calculate latent representations
	# 		self.act1 = self.FID_model.predict(self.fid_train_images)
	# 		self.act2 = self.FID_model.predict(preds)
	# 		self.eval_FID()


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

