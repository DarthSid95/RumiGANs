from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import glob
from absl import flags
import csv

from scipy import io as sio

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import PdfPages

import tensorflow_datasets as tfds
# import tensorflow_datasets as tfds


### Need to prevent tfds downloads bugging out? check
import urllib3
urllib3.disable_warnings()


FLAGS = flags.FLAGS

'''***********************************************************************************
********** Base Data Loading Ops *****************************************************
***********************************************************************************'''
class GAN_DATA_ops:

	def __init__(self):
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data)'
		self.noise_mean = 0.0
		self.noise_stddev = 1.00
		#Default Number of repetitions of a dataset in tf.dataset mapping
		self.reps = 1

		if self.data == 'g1':
			self.MIN = 0
			self.MAX = 1
			self.noise_dims = 1
			self.output_size = 1
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		elif self.data == 'g2':
			self.MIN = -1
			self.MAX = 1.2
			self.noise_dims = 2
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		elif self.data == 'gN':
			self.MIN = 0
			self.MAX = 1
			self.noise_dims = 100
			self.output_size = self.GaussN
			self.noise_stddev = 1.0
			self.noise_mean = 0.0
		elif self.data == 'gmm8':
			self.noise_dims = 100
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		elif self.data == 'gmm2':
			self.MIN = -0.5
			self.MAX = 10.5
			self.noise_dims = 100
			self.noise_mean = 0.0
			self.output_size = 2
			self.noise_stddev = 1.
		else: 
			self.noise_dims = 100
			if self.data == 'celeba':
				self.output_size = self.out_size
			elif self.data == 'mnist':
				self.output_size = 28
			elif self.data == 'cifar10':
				self.output_size = 32
			if self.data == 'comma':
				self.output_H = 120
				self.output_W = 240

		# self.testcase = testcase 
		# self.number = number

	def mnist_loader(self):
		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
		# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
		train_images = train_images.reshape(train_images.shape[0],self.output_size, self.output_size, 1).astype('float32')
		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		# train_images = (train_images - 0.) / 255.0
		test_images = test_images.reshape(test_images.shape[0],self.output_size, self.output_size, 1).astype('float32')
		self.test_images = (test_images - 127.5) / 127.5
		# self.test_images = (test_images - 0.) / 255.0

		return train_images, train_labels, test_images, test_labels


	def fmnist_loader(self):
		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
		train_images = train_images.reshape(train_images.shape[0],self.output_size, self.output_size, 1).astype('float32')
		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		test_images = test_images.reshape(test_images.shape[0],self.output_size, self.output_size, 1).astype('float32')
		test_labels = test_labels.reshape(test_images.shape[0], 1).astype('float32')
		test_images = (test_images - 127.5) / 127.5


		return train_images, train_labels, test_images, test_labels


	def celeba_loader(self):
		if self.colab:
			try:
				with open("data/CelebA/Colab_CelebA_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('/content/colab_data_faces/img_align_celeba/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/CelebA/Colab_CelebA_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		else:
			try:
				with open("data/CelebA/CelebA_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('data/CelebA/img_align_celeba/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/CelebA/CelebA_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		train_images = np.expand_dims(np.array(true_files),axis=1)

		attr_file = 'data/CelebA/list_attr_celeba.csv'

		with open(attr_file,'r') as a_f:
			data_iter = csv.reader(a_f,delimiter = ',',quotechar = '"')
			data = [data for data in data_iter]
		# print(data,len(data))
		label_array = np.asarray(data)

		return train_images, label_array


	def cifar10_loader(self):

		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
		train_images = train_images.reshape(train_images.shape[0],self.output_size, self.output_size, 3).astype('float32')
		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		test_images = test_images.reshape(test_images.shape[0],self.output_size, self.output_size, 3).astype('float32')
		test_labels = test_labels.reshape(test_images.shape[0], 1).astype('float32')
		test_images = (test_images - 127.5) / 127.5

		return train_images, train_labels, test_images, test_labels


	def lsun_loader(self):
		if self.colab:
			try:
				train_images = tfds.load('lsun/bedroom', split='train', data_dir='data/LSUN', batch_size=None, shuffle_files=True ,download=False, as_supervised=False, decoders=None, read_config=None,with_info=False, builder_kwargs=None, download_and_prepare_kwargs=None,as_dataset_kwargs=None, try_gcs=False)

				# with open("data/LSUN/Colab_LSUN_Names.txt","r") as names:
					# true_files = np.array([line.rstrip() for line in names])
					# print("Data File Found. Reading filenames")
			except:
				tfds.disable_progress_bar()
				train_images = tfds.load('lsun/bedroom', split='train', data_dir='data/LSUN', batch_size=None, shuffle_files=True ,download=True, as_supervised=False, decoders=None, read_config=None,with_info=False, builder_kwargs=None, download_and_prepare_kwargs=None,as_dataset_kwargs=None, try_gcs=False)
				# print("BUGS")
				# true_files = sorted(glob.glob('/content/LSUN/bedroom_train/*.jpg'))
				# print("Data File Created. Saving filenames")
				# with open("data/LSUN/Colab_LSUN_Names.txt","w") as names:
				# 	for name in true_files:
				# 		names.write(str(name)+'\n')
		else:
			try:
				with open("data/LSUN/LSUN_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('data/LSUN/bedroom_train/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/LSUN/LSUN_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')

		return train_images

'''
GAN_DATA functions are specific to the topic, ELeGANt, RumiGAN, PRDeep or DCS. Data reading and dataset making functions per data, with init having some specifics generic to all, such as printing instructions, noise params. etc.
'''
'''***********************************************************************************
********** GAN_DATA_Baseline *********************************************************
***********************************************************************************'''
class GAN_DATA_Base(GAN_DATA_ops):

	def __init__(self):#,data,testcase,number,out_size):
		# self.gen_func = 'self.gen_func_'+data+'()'
		# self.dataset_func = 'self.dataset_'+data+'(self.train_data)'
		self.noise_mean = 0.0
		self.noise_stddev = 1.00
		GAN_DATA_ops.__init__(self)#,data,testcase,number,out_size)

	def gen_func_mnist(self):
		# self.output_size = int(28)

		train_images, train_labels, test_images, test_labels = self.mnist_loader()

		zero = train_labels == 0
		one = train_labels == 1
		two  = train_labels == 2
		three  = train_labels == 3
		four  = train_labels == 4
		five  = train_labels == 5
		six  = train_labels == 6
		seven  = train_labels == 7
		eight = train_labels == 8
		nine = train_labels == 9

		self.fid_train_images = train_images

		if self.testcase == 'single':	
			train_images = train_images[np.where(train_labels == self.number)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'few':	
			self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
			train_images = train_images[np.where(train_labels == self.number)[0]][0:self.num_few]
			self.fid_train_images_few = train_images
			#train_images[np.where(train_labels == self.number)[0][0:500]]
		if self.testcase == 'even':
			train_images = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'odd':
			train_images = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'sharp':
			train_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			self.fid_train_images = train_images

		self.reps = int(60000.0/train_images.shape[0])
		return train_images

	def dataset_mnist(self,train_data,batch_size):

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		if self.loss == 'deq':
			if self.latent_kind == 'DCT':# or self.latent_kind == 'Cycle':
				train_dataset = train_dataset.map(DCT_compression_function, num_parallel_calls=int(self.num_parallel_calls))
		if self.testcase == 'single' or self.testcase == 'few':
			train_dataset = train_dataset.repeat(self.reps-1)
		train_dataset = train_dataset.shuffle(50000)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(10)
		return train_dataset

	def gen_func_celeba(self):

		train_images, data_array = self.celeba_loader()
		# print(data_array,data_array.shape)
		tags = data_array[0,:] # print to find which col to pull for what
		
		
		
		
		# print(gender,gender.shape)
		gender = data_array[1:,21]
		male = gender == '1'
		male = male.astype('uint8')

		bald_labels = data_array[1:,5]
		bald = bald_labels == '1'
		bald = bald.astype('uint8')

		hat_labels = data_array[1:,-5]
		hat = hat_labels == '1'
		hat = hat.astype('uint8')

		mustache_labels = data_array[1:,23]
		hustache = mustache_labels == '1'
		hustache = hustache.astype('uint8')

		self.fid_train_images = train_images

		# if self.testcase == 'single':
		# 	self.fid_train_images = train_images[np.where(male == 0)]
		# 	train_images = train_images[np.where(male == 0)]
		if self.testcase == 'female':
			train_images = train_images[np.where(male == 0)]
			self.fid_train_images = train_images
		if self.testcase == 'male':
			train_images = train_images[np.where(male == 1)]
			self.fid_train_images = train_images
		if self.testcase == 'fewfemale':
			self.fid_train_images = train_images[np.where(male == 0)]
			train_images = np.repeat(train_images[np.where(male == 0)][0:self.num_few],20,axis = 0)
		if self.testcase == 'fewmale':
			self.fid_train_images = train_images[np.where(male == 1)]
			train_images = np.repeat(train_images[np.where(male == 0)][0:self.num_few],20,axis = 0)
		if self.testcase == 'bald':
			self.fid_train_images = train_images[np.where(bald == 1)]
			train_images = np.repeat(train_images[np.where(bald == 1)],20,axis = 0)
		if self.testcase == 'hat':
			self.fid_train_images = train_images[np.where(hat == 1)]
			train_images = np.repeat(train_images[np.where(hat == 1)],20,axis = 0)

		return train_images

	def dataset_celeba(self,train_data,batch_size):	
		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[self.output_size,self.output_size])

				# This will convert to float values in [-1, 1]
				image = tf.divide(image,255.0)
				# image = tf.subtract(image,127.0)
				# image = tf.divide(image,127.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.map(data_reader_faces, num_parallel_calls=int(self.num_parallel_calls))
		# train_dataset = train_dataset.map(DCT_compression_function, num_parallel_calls=int(self.num_parallel_calls))
		train_dataset = train_dataset.shuffle(500)
		train_dataset = train_dataset.batch(batch_size, drop_remainder = True)
		train_dataset = train_dataset.prefetch(15)
		return train_dataset

	def gen_func_cifar10(self):
		# self.output_size = int(28)

		train_images, train_labels, test_images, test_labels = self.cifar10_loader()

		zero = train_labels == 0
		one = train_labels == 1
		two  = train_labels == 2
		three  = train_labels == 3
		four  = train_labels == 4
		five  = train_labels == 5
		six  = train_labels == 6
		seven  = train_labels == 7
		eight = train_labels == 8
		nine = train_labels == 9
		# print(train_labels)
		# exit(0)
		# CLASSES: airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck
		if self.testcase == 'few':
			self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
			train_images = train_images[np.where(train_labels == self.number)[0]][0:self.num_few]
			self.reps = int(50000.0/train_images.shape[0])
		if self.testcase == 'single':
			train_images = train_images[np.where(train_labels == self.number)[0]]
			self.fid_train_images = train_images
			# mean_num = np.mean(train_images,axis = 0)
			# print(mean_num.shape)
			# self.save_paper(mean_num[:,:,0])
			self.reps = int(60000/train_images.shape[0])+1
		if self.testcase == 'even':
			train_images = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'odd':
			train_images = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'animals':
			train_images = train_images[np.where(np.any([two, three, four, five, six, seven],axis=0))[0]]
			self.fid_train_images = train_images


		return train_images

	def dataset_cifar10(self,train_data,batch_size):
		
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		if self.testcase == 'single' or self.testcase == 'few' or self.testcase == 'bald':
			train_dataset = train_dataset.repeat(self.reps-1)
		train_dataset = train_dataset.shuffle(400)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset


	def gen_func_g1(self):
		self.MIN = -3.5
		self.MAX = 10.5
		# g1 = tfp.distributions.TruncatedNormal(loc=7.0, scale=1.0, low=-20., high=20.)
		g1 = tfp.distributions.TruncatedNormal(loc=1.0, scale=0.50, low=-20., high=20.)
		return g1.sample([800*self.batch_size, 1])
		# return tf.random.normal([1000*self.batch_size, 1], mean = 8.0, stddev = 1.)


	def dataset_g1(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset

	def gen_func_gN(self):
		# g1 = tfp.distributions.TruncatedNormal(loc=0., scale=0.1, low=-1., high=1.)
		# return g1.sample([100*self.batch_size, self.output_size])
		# return tf.random.normal([200*self.batch_size, self.GaussN], mean = 2.5*np.ones((1,self.output_size)), stddev = 0.8*np.ones((1,self.output_size)))
		return tf.random.normal([200*self.batch_size, self.GaussN], mean = 0.7*np.ones((1,self.output_size)), stddev = 0.2*np.ones((1,self.output_size)))

	def dataset_gN(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset

	def gen_func_gmm2(self):
		self.MIN = -1
		self.MAX = 12
		tfd = tfp.distributions
		probs = [0.5, 0.25, 0.25]
		locs = [[1.2],[6.0], [8.7]]
		# stddev_scale = [.04, .04, .04, .04, .04, .04, .04, .04]
		stddev_scale = [0.21, 0.2, 0.21]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# covs = [ [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]]   ]

		gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		return gmm.sample(sample_shape=(int(1000*self.batch_size.numpy())))

	def dataset_gmm2(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(10)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(10)
		return train_dataset

	def gen_func_g2(self):
		self.MIN = -2.2
		self.MAX = 2.
		return tf.random.normal([500*self.batch_size.numpy(), 2], mean =np.array([1.0,1.0]), stddev = np.array([0.20,0.20]))

	def dataset_g2(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset

	def gen_func_gmm8(self):
		tfd = tfp.distributions
		probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
		## Cirlce
		# locs = [[1., 0.], [0., 1.], [-1.,0.], [0.,-1.], [1*0.7071, 1*0.7071], [-1*0.7071, 1*0.7071], [1*0.7071, -1*0.7071], [-1*0.7071, -1*0.7071] ]
		# self.MIN = -1.3 #-1.3 for circle, 0 for pattern
		# self.MAX = 1.3 # +1.3 for cicle , 1 for pattern

		## ?
		# locs = [[0.25, 0.], [0., 0.25], [-0.25,0.], [0.,-0.25], [0.25*0.7071, 0.5*0.7071], [-0.25*0.7071, 0.25*0.7071], [0.25*0.7071, -0.25*0.7071], [-0.25*0.7071, -0.25*0.7071] ]

		## random
		# locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.75*0.7071, 0.75*0.7071], [0.25*0.7071, 0.75*0.7071], [0.75*0.7071, 0.25*0.7071], [0.25*0.7071, 0.25*0.7071] ]

		## Pattern
		locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.5*1.7071, 0.5*1.7071], [0.5*0.2929, 0.5*1.7071], [0.5*1.7071, 0.5*0.2929], [0.5*0.2929, 0.5*0.2929] ]
		self.MIN = -0. #-1.3 for circle, 0 for pattern
		self.MAX = 1.0 # +1.3 for cicle , 1 for pattern

		# stddev_scale = [.04, .04, .04, .04, .04, .04, .04, .04]
		stddev_scale = [.03, .03, .03, .03, .03, .03, .03, .03]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# covs = [ [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]]   ]

		gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		return gmm.sample(sample_shape=(int(500*self.batch_size.numpy())))

	def dataset_gmm8(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset


	def gen_func_comma(self):
		import h5py
		try:
			self.h5 = h5py.File('data/CommaAI/CommaAI_dataset.h5', 'r')
		except:
			try:
				with open("data/CommaAI/Comma_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
				with open("data/CommaAI/Comma_Logs.txt","r") as names:
					log_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('data/CommaAI/camera/*.h5'))
				log_files = sorted(glob.glob('data/CommaAI/log/*.h5'))
				print("Data File Created. Saving filenames")
				with open("data/CommaAI/Comma_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
				with open("data/CommaAI/Comma_Logs.txt","w") as logs:
					for log in log_files:
						logs.write(str(log)+'\n')
			
			'''
			Code Courtesy Comma AI research: 
			https://github.com/commaai/research/blob/master/dask_generator.py 
			'''
			angle = []  # steering angle of the car
			speed = []  # steering angle of the car
			hdf5_camera = []  # the camera hdf5 files need to continue open
			c5x = []
			filters = []
			x_vec = []
			lastidx = 0
			time_len = 1
			#Need to read out data and steeding angles
			print(true_files)
			print(log_files)
			for cword, tword in zip(true_files[0:5], log_files[0:5]):
				print(cword,tword)
				try:
					with h5py.File(tword, "r") as t5:
						c5 = h5py.File(cword, "r")
						hdf5_camera.append(c5)
						x = c5["X"]
						c5x.append((lastidx, lastidx+x.shape[0], x))


						speed_value = t5["speed"][:]
						steering_angle = t5["steering_angle"][:]
						idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
						xs = x[:]
						if lastidx == 0:
							x_vec = np.array(xs)
						else:
							x_vec = np.concatenate([x_vec, xs], axis = 0)
						# x_vec.append(xs)
						angle.append(steering_angle[idxs])
						speed.append(speed_value[idxs])

						goods = np.abs(angle[-1]) <= 200

						filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
						lastidx += goods.shape[0]
						# check for mismatched length bug
						print("x {} | t {} | f {} | xv {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0], xs.shape[0]))
						if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
							raise Exception("bad shape")
						print(x.shape, steering_angle.shape, goods.shape)

				except IOError:
					import traceback
					traceback.print_exc()
					print('failed to open'), tword

			

			angle = np.concatenate(angle, axis=0)
			speed = np.concatenate(speed, axis=0)
			filters = np.concatenate(filters, axis=0).ravel()

			true_frames = x_vec[filters]
			true_angle = angle[filters]
			true_speed = speed[filters]

			hf = h5py.File('CommaAI_dataset.h5', 'w')
			hf.create_dataset('frames', data=true_frames)
			hf.create_dataset('angle', data=true_angle)
			hf.create_dataset('speed', data=true_speed)
			hf.close()

			self.h5 = h5py.File('data/CommaAI/CommaAI_dataset.h5', 'r')

			print('training on {}/{} examples'.format(filters.shape[0], angle.shape[0]))
			print(angle, filters, speed)
			print(angle.shape, filters.shape, speed.shape)

			print('training on {} examples, {} angles'.format(true_frames.shape[0], true_angle.shape[0]))
		
		#### Create a sequences for randomly selecting batch from the h5. 

		# a list of number of batches possible 
		self.num_h5_batches = int(self.h5['angle'].shape[0] / self.batch_size)
		self.sel_vec = np.arange(self.h5['angle'].shape[0])
		print(self.sel_vec)
		np.random.shuffle(self.sel_vec)
		print(self.sel_vec)
		# print(self.h5['frames'][self.sel_vec])
		return self.h5['angle']#true_frames#c5x, angle, speed, filters, hdf5_camera

	def dataset_comma(self,train_data,batch_size):
		class generator:
			def __init__(self,h5,sel_vec):
				self.h5_var = h5
				self.sel_v = sel_vec
				np.random.shuffle(self.sel_v)

			def __call__(self):
				for sel in self.sel_v:
					yield self.h5_var['frames'][sel]

		def resize_func(im):
			im = tf.transpose(im, [1, 2, 0])
			im = tf.image.resize(im, [self.output_H, self.output_W])
			im = tf.divide(im, tf.constant(255, dtype = 'float32'))
			return im

		train_dataset = tf.data.Dataset.from_generator(generator(self.h5, self.sel_vec), (tf.uint8), (tf.TensorShape([ 3, 160, 320]))) 
		train_dataset = train_dataset.map(resize_func, num_parallel_calls = int(self.num_parallel_calls))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		# train_dataset = train_dataset.cache(filename=self.cache_loc)
		return train_dataset


'''************************************************************************************
********** GAN_DATA_RumiGAN ***********************************************************
***********************************************************************************'''
class GAN_DATA_RumiGAN(GAN_DATA_ops):

	def __init__(self):#,data,testcase,number,out_size):
		GAN_DATA_ops.__init__(self)#,data,testcase,number,out_size)

	def gen_func_mnist(self):
		train_images, train_labels, test_images, test_labels = self.mnist_loader()

		zero = train_labels == 0
		one = train_labels == 1
		two  = train_labels == 2
		three  = train_labels == 3
		four  = train_labels == 4
		five  = train_labels == 5
		six  = train_labels == 6
		seven  = train_labels == 7
		eight = train_labels == 8
		nine = train_labels == 9

		# print(sum(one), sum(two), sum(four), sum(five), sum(seven), sum(nine))
		# print( sum(np.any([one, two, four, five, seven, nine],axis=0)))
		# print(x)

		if self.testcase == 'single':	
			true_images = train_images[np.where(train_labels == self.number)[0]]
			false_images = train_images[np.where(train_labels != self.number)[0]]
			self.fid_train_images = train_images
			self.fid_others = false_images
		if self.testcase == 'few':	
			true_images = train_images[np.where(train_labels == self.number)[0]][0:self.num_few]
			false_images = train_images[np.where(train_labels != self.number)[0]]
			self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
			self.fid_train_images_few = true_images
			self.fid_train_images = true_images
			self.fid_others = false_images
			#train_images[np.where(train_labels == self.number)[0][0:500]]
		if self.testcase == 'even':
			true_images = train_images[np.where(train_labels%2 == 0)[0]]
			false_images = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_train_images = true_images
			self.fid_others = false_images
		if self.testcase == 'odd':
			true_images = train_images[np.where(train_labels%2 != 0)[0]]
			false_images = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_train_images = true_images
			self.fid_others = false_images
		if self.testcase == 'sharp':
			true_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			false_images = train_images[np.where(np.any([zero, two, three, six, eight, nine],axis=0))[0]]
			self.fid_train_images = true_images
			self.fid_others = false_images
		if self.testcase == 'sharpSVHN':
			SVHN_train_data = sio.loadmat('data/SVHN/train_32x32.mat')
			# access to the dict
			train_images_SVHN = tf.image.resize(tf.image.rgb_to_grayscale(tf.transpose(tf.cast(SVHN_train_data['X'],dtype='float32'),[3,0,1,2])),[self.output_size,self.output_size]).numpy()
			train_images_SVHN = (train_images_SVHN - 0.) / 255.0
			train_labels_SVHN = SVHN_train_data['y']
			SVHNzero = train_labels_SVHN == 0
			SVHNone = train_labels_SVHN == 1
			SVHNtwo  = train_labels_SVHN == 2
			SVHNthree  = train_labels_SVHN == 3
			SVHNfour  = train_labels_SVHN == 4
			SVHNfive  = train_labels_SVHN == 5
			SVHNsix  = train_labels_SVHN == 6
			SVHNseven  = train_labels_SVHN == 7
			SVHNeight = train_labels_SVHN == 8
			SVHNnine = train_labels_SVHN == 9

			true_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			self.fid_train_images = true_images
			false_images = train_images_SVHN[np.where(np.any([SVHNzero, SVHNtwo, SVHNthree, SVHNsix, SVHNeight, SVHNnine],axis=0))[0]]
			self.fid_others = false_images
		if self.testcase == 'SVHN':
			SVHN_train_data = sio.loadmat('data/SVHN/train_32x32.mat')
			# access to the dict
			train_images_SVHN = tf.image.resize(tf.image.rgb_to_grayscale(tf.transpose(tf.cast(SVHN_train_data['X'],dtype='float32'),[3,0,1,2])),[self.output_size,self.output_size]).numpy()
			train_images_SVHN = (train_images_SVHN - 0.) / 255.0
			
			true_images = train_images
			self.fid_train_images = true_images
			false_images = train_images_SVHN
			self.fid_others = false_images

		self.ratio = true_images.shape[0] / float(false_images.shape[0])

		return true_images, false_images

	def dataset_mnist(self,train_data_pos, train_data_neg, batch_size):

		train_dataset_pos = tf.data.Dataset.from_tensor_slices((train_data_pos))
		if self.ratio < 1 :
			reps = np.ceil(1/float(self.ratio))
			train_dataset_pos = train_dataset_pos.repeat(reps)
		train_dataset_pos = train_dataset_pos.shuffle(50000)
		train_dataset_pos = train_dataset_pos.batch(batch_size, drop_remainder = True)
		train_dataset_pos = train_dataset_pos.prefetch(5)

		train_dataset_neg = tf.data.Dataset.from_tensor_slices((train_data_neg))
		if self.ratio >= 1 :
			reps = np.ceil(self.ratio) 
			train_dataset_neg = train_dataset_neg.repeat(reps)
		train_dataset_neg = train_dataset_neg.shuffle(50000)
		train_dataset_neg = train_dataset_neg.batch(batch_size, drop_remainder = True)
		train_dataset_neg = train_dataset_neg.prefetch(5)
		return train_dataset_pos, train_dataset_neg

	def gen_func_cifar10(self):
		# self.output_size = int(28)
		train_images, train_labels, test_images, test_labels = self.cifar10_loader()

		zero = train_labels == 0
		one = train_labels == 1
		two  = train_labels == 2
		three  = train_labels == 3
		four  = train_labels == 4
		five  = train_labels == 5
		six  = train_labels == 6
		seven  = train_labels == 7
		eight = train_labels == 8
		nine = train_labels == 9


		# print(train_labels)
		# exit(0)
		# CLASSES: airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck
		if self.testcase == 'few':
			self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
			true_images = train_images[np.where(train_labels == self.number)[0]][0:self.num_few]
			false_images = train_images[np.where(train_labels != self.number)[0]]
		if self.testcase == 'single':
			true_images = train_images[np.where(train_labels == self.number)[0]]
			self.fid_train_images = true_images
			false_images = train_images[np.where(train_labels != self.number)[0]]
		if self.testcase == 'animals':
			true_images = train_images[np.where(np.any([two, three, four, five, six, seven],axis=0))[0]]
			self.fid_train_images = true_images
			false_images = train_images[np.where(np.any([zero, one, eight, nine],axis=0))[0]]


		self.ratio = true_images.shape[0] / float(false_images.shape[0])

		return true_images, false_images

	def dataset_cifar10(self,train_data_pos, train_data_neg, batch_size):


		train_dataset_pos = tf.data.Dataset.from_tensor_slices((train_data_pos))
		if self.ratio < 1 :
			reps = np.ceil(1/float(self.ratio))
			train_dataset_pos = train_dataset_pos.repeat(reps)
		train_dataset_pos = train_dataset_pos.shuffle(50000)
		train_dataset_pos = train_dataset_pos.batch(batch_size, drop_remainder = True)
		train_dataset_pos = train_dataset_pos.prefetch(5)

		train_dataset_neg = tf.data.Dataset.from_tensor_slices((train_data_neg))
		if self.ratio >= 1 :
			reps = np.ceil(self.ratio) 
			train_dataset_neg = train_dataset_neg.repeat(reps)
		train_dataset_neg = train_dataset_neg.shuffle(50000)
		train_dataset_neg = train_dataset_neg.batch(batch_size, drop_remainder = True)
		train_dataset_neg = train_dataset_neg.prefetch(5)
		return train_dataset_pos, train_dataset_neg


	def gen_func_wce(self):

		try:
			with open("data/WCE/Iakovidis_WCE_Dataset/train_list.txt","r") as names:
				true_files = np.array(['data/WCE/Iakovidis_WCE_Dataset/'+line.rstrip().split('sorted/')[1] for line in names])
				print("Data File Found. Reading filenames")
		except:
			true_files = sorted(glob.glob('data/WCE/Iakovidis_WCE_Dataset/**.png'))
			print("Data File Created. Saving filenames")
			with open("data/WCE/Iakovidis_WCE_Dataset/train_list.txt","w") as names:
				for name in true_files:
					names.write(str(name)+'\n')
		if self.colab:
			try:
				train_images = np.load('data/WCE/Iakovidis_WCE_Dataset/WCE_images.npy')
				print("Successfully Loaded WCE numpy file")
			except:
				names_list = np.expand_dims(np.array(true_files),axis=1)
				train_images = np.zeros([len(names_list),self.output_size,self.output_size,3])
				with tf.device("/CPU"):
					for i,filename in enumerate(names_list):
						# print(i,filename)
						image_string = tf.io.read_file(filename[0])
						image = tf.image.decode_jpeg(image_string, channels=3)
						image.set_shape([360,360,3])
						image = tf.image.resize(image,[self.output_size,self.output_size])
						image = tf.divide(image,255.0)
						train_images[i] = image
					print(train_images.shape)
					np.save('data/WCE/Iakovidis_WCE_Dataset/WCE_images.npy',train_images)
		else:
			train_images = np.expand_dims(np.array(true_files),axis=1)

		abnormal_cats = ['Angioectasias', 'Apthae', 'Bleeding', 'ChylousCysts', 'Lymphangectasias', 'Polypoids', 'Stenoses', 'Ulcers', 'VillousOedemas']
		base_cats = 'Normal'

		with open("data/WCE/Iakovidis_WCE_Dataset/train_list.txt","r") as names:
				attr = np.array([line.rstrip().split('sorted/')[1].split('/')[0] for line in names])
				print("Data File Found. Reading filenames")

		abnormal = attr == abnormal_cats[self.number]

		if self.testcase == 'single':
			true_images = train_images[np.where(abnormal == 1)]
			false_images = train_images[np.where(abnormal == 0)]
			self.fid_train_images = train_images[np.where(abnormal == 1)]

		self.ratio = true_images.shape[0] / float(false_images.shape[0])


		# print(true_images, false_images)
		return true_images, false_images

	def dataset_wce(self,train_data_pos, train_data_neg, batch_size):
		
		def flip_func(image):
			with tf.device('/CPU'):
				image = tf.image.flip_left_right(image)
				return image

		def rot_func(image):
			with tf.device('/CPU'):
				image = tf.image.rot90(image)
				return image

		def data_reader(filename):
			with tf.device('/CPU'):
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([360,360,3])
				image = tf.image.resize(image,[self.output_size,self.output_size])

				# This will convert to float values in [0, 1]
				image = tf.divide(image,255.0)


				# image = tf.image.convert_image_dtype(image, tf.float32)
				# image = tf.divide(image,255.0)
			return image

		train_dataset_pos = tf.data.Dataset.from_tensor_slices((train_data_pos))
		if self.ratio < 1 :
			reps = np.ceil(1/float(self.ratio))
			train_dataset_pos = train_dataset_pos.repeat(reps)
		if not self.colab:
			train_dataset_pos = train_dataset_pos.map(data_reader, num_parallel_calls=int(self.num_parallel_calls))
		images_flip = train_dataset_pos.map(flip_func, num_parallel_calls=int(self.num_parallel_calls))
		images_rot = train_dataset_pos.map(rot_func, num_parallel_calls=int(self.num_parallel_calls))
		images_fliprot = images_flip.map(rot_func, num_parallel_calls=int(self.num_parallel_calls))
		train_dataset_pos = train_dataset_pos.concatenate(images_flip).concatenate(images_rot).concatenate(images_fliprot)
		train_dataset_pos = train_dataset_pos.shuffle(4)
		train_dataset_pos = train_dataset_pos.batch(batch_size)
		train_dataset_pos = train_dataset_pos.prefetch(5)

		train_dataset_neg = tf.data.Dataset.from_tensor_slices((train_data_neg))
		if self.ratio >= 1 :
			reps = np.ceil(self.ratio) 
			train_dataset_neg = train_dataset_neg.repeat(reps)
		if not self.colab:
			train_dataset_neg = train_dataset_neg.map(data_reader, num_parallel_calls=int(self.num_parallel_calls))
		train_dataset_neg = train_dataset_neg.shuffle(10000)
		images_flip = train_dataset_neg.map(flip_func, num_parallel_calls=int(self.num_parallel_calls))
		images_rot = train_dataset_neg.map(rot_func, num_parallel_calls=int(self.num_parallel_calls))
		images_fliprot = images_flip.map(rot_func, num_parallel_calls=int(self.num_parallel_calls))
		train_dataset_neg = train_dataset_neg.concatenate(images_flip).concatenate(images_rot).concatenate(images_fliprot)
		train_dataset_neg = train_dataset_neg.batch(batch_size)
		train_dataset_neg = train_dataset_neg.prefetch(5)
		return train_dataset_pos, train_dataset_neg

	def gen_func_celeba(self):

		train_images, data_array = self.celeba_loader()
		# print(data_array,data_array.shape)
		tags = data_array[0,:] # print to find which col to pull for what
		# print(tags)
		# print(XXX)
		gender_labels = data_array[1:,21]
		Eyeglass_labels = data_array[1:,16]
		Mush_labels = data_array[1:,23]
		bald_labels = data_array[1:,5]
		# print(gender,gender.shape)
		male = gender_labels == '1'
		Eyeglasses = Eyeglass_labels == '1'
		Mustache = Mush_labels == '1'
		bald = bald_labels == '1'

		hat_labels = data_array[1:,-5]
		hat = hat_labels == '1'
		hat = hat.astype('uint8')



		if self.testcase == 'fewfemale':
			true_images = train_images[np.where(male == 0)][0:self.num_few]
			false_images = train_images[np.where(male == 1)]
			self.fid_train_images = train_images[np.where(male == 0)]
		if self.testcase == 'fewmale':
			true_images = train_images[np.where(male == 1)][0:self.num_few]
			false_images = train_images[np.where(male == 0)]
			self.fid_train_images = train_images[np.where(male == 1)]
		if self.testcase == 'female':
			true_images = train_images[np.where(male == 0)]
			false_images = train_images[np.where(male == 1)]
			self.fid_train_images = train_images[np.where(male == 0)]
		if self.testcase == 'male':
			true_images = train_images[np.where(male == 1)]
			false_images = train_images[np.where(male == 0)]
			self.fid_train_images = train_images[np.where(male == 1)]
		if self.testcase == 'bald':
			true_images = train_images[np.where(bald == 1)]
			false_images = train_images[np.where(bald == 0)]
			self.fid_train_images = train_images[np.where(bald == 1)]
		if self.testcase == 'cifar10':
			true_images = train_images
			self.fid_train_images = true_images
			(false_images, false_labels), (false_test_images, false_test_labels) = tf.keras.datasets.cifar10.load_data()
			false_images = false_images.reshape(false_images.shape[0], self.output_size, self.output_size, 3).astype('float32')
			false_labels = false_labels.reshape(false_labels.shape[0], 1).astype('float32')
			false_images = (false_images - 0.) / 255.0

		if self.testcase == 'hat':
			true_images = train_images[np.where(hat == 1)]
			false_images = train_images[np.where(hat == 0)]
			self.fid_train_images = train_images[np.where(hat == 1)]
		if self.testcase == 'custom':
			true_images = train_images[np.where(np.all([Eyeglasses,male,Mustache],axis = 0).astype('uint8') == 1)]
			false_images = train_images[np.where(np.all([Eyeglasses,male,Mustache],axis = 0).astype('uint8') == 0)]
			self.fid_train_images = true_images

		self.ratio = true_images.shape[0] / float(false_images.shape[0])

		return true_images, false_images


	def dataset_celeba(self,train_data_pos, train_data_neg, batch_size):
		
		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[self.output_size,self.output_size])

				# This will convert to float values in [-1, 1]
				image = tf.divide(image,255.0)
				# image = tf.subtract(image,127.0)
				# image = tf.divide(image,127.0)
				# image = tf.image.convert_image_dtype(image, tf.float32)
				# image = tf.divide(image,255.0)
			return image

		train_dataset_pos = tf.data.Dataset.from_tensor_slices((train_data_pos))
		if self.ratio < 1 :
			reps = np.ceil(1/float(self.ratio))
			train_dataset_pos = train_dataset_pos.repeat(reps)
		# if not self.colab:
		train_dataset_pos = train_dataset_pos.map(data_reader_faces, num_parallel_calls=int(self.num_parallel_calls))
		train_dataset_pos = train_dataset_pos.shuffle(500)
		train_dataset_pos = train_dataset_pos.batch(batch_size, drop_remainder = True)
		train_dataset_pos = train_dataset_pos.prefetch(5)

		train_dataset_neg = tf.data.Dataset.from_tensor_slices((train_data_neg))
		if self.ratio >= 1 :
			reps = np.ceil(self.ratio) 
			train_dataset_neg = train_dataset_neg.repeat(reps)
		if self.testcase != 'cifar10':
			train_dataset_neg = train_dataset_neg.map(data_reader_faces, num_parallel_calls=int(self.num_parallel_calls))
		train_dataset_neg = train_dataset_neg.shuffle(500)
		train_dataset_neg = train_dataset_neg.batch(batch_size)
		train_dataset_neg = train_dataset_neg.prefetch(5)
		return train_dataset_pos, train_dataset_neg


	def gen_func_g1(self):
		self.MIN = -0.2
		self.MAX = 2.0
		return tf.random.normal([1000*self.batch_size, 1], mean =1.0, stddev = 0.5), tf.random.normal([1000*self.batch_size, 1], mean =0.75, stddev = 0.2)

	def dataset_g1(self, train_data_pos, train_data_neg, batch_size):
		train_dataset_pos = tf.data.Dataset.from_tensor_slices((train_data_pos))
		train_dataset_pos = train_dataset_pos.shuffle(5000)
		train_dataset_pos = train_dataset_pos.batch(batch_size)
		train_dataset_pos = train_dataset_pos.prefetch(5)

		train_dataset_neg = tf.data.Dataset.from_tensor_slices((train_data_neg))
		train_dataset_neg = train_dataset_neg.shuffle(5000)
		train_dataset_neg = train_dataset_neg.batch(batch_size)
		train_dataset_neg = train_dataset_neg.prefetch(5)

		return train_dataset_pos, train_dataset_neg

	def gen_func_u1(self):
		self.MIN = -0.2
		self.MAX = 1.5
		train_data_pos = tf.random.uniform([100*self.batch_size, 1], minval =0.50, maxval = 0.9, seed = FLAGS.seed)
		train_data_neg = np.concatenate((tf.random.uniform([50*self.batch_size, 1], minval =0.2, maxval = 0.65, seed = FLAGS.seed),tf.random.uniform([50*self.batch_size, 1], minval =0.85, maxval = 1., seed = FLAGS.seed)), axis = 0)

		return train_data_pos, train_data_neg


	def dataset_u1(self, train_data_pos, train_data_neg, batch_size):
		train_dataset_pos = tf.data.Dataset.from_tensor_slices((train_data_pos))
		train_dataset_pos = train_dataset_pos.shuffle(50000)
		train_dataset_pos = train_dataset_pos.batch(batch_size)
		train_dataset_pos = train_dataset_pos.prefetch(50)

		train_dataset_neg = tf.data.Dataset.from_tensor_slices((train_data_neg))
		train_dataset_neg = train_dataset_neg.shuffle(50000)
		train_dataset_neg = train_dataset_neg.batch(batch_size)
		train_dataset_neg = train_dataset_neg.prefetch(50)

		return train_dataset_pos, train_dataset_neg

	def gen_func_g2(self):
		self.MIN = -0.6
		self.MAX = 0.9
		# probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
		# locs = [[1., 0.5], [0.5, 1.], [0.,0.5], [0.5,0.], [0.5*1.7071, 0.5*1.7071], [0.5*0.2929, 0.5*1.7071], [0.5*1.7071, 0.5*0.2929], [0.5*0.2929, 0.5*0.2929] ]
		# locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.5*1.7071, 0.5*1.7071], [0.5*0.2929, 0.5*1.7071], [0.5*1.7071, 0.5*0.2929], [0.5*0.2929, 0.5*0.2929] ]

		# stddev_scale = [.03, .03, .03, .03, .03, .03, .03, .03]
		# gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		# # data_pos = tf.random.normal([800*self.batch_size, 2], mean =np.array([1.50,1.50]), stddev = np.array([0.5,0.5]))
		# data_pos = gmm.sample(sample_shape=(int(800*self.batch_size.numpy())))

		# data_neg = tf.random.normal([800*self.batch_size, 2], mean =np.array([0.75,0.5]), stddev = np.array([0.1,0.1]))
		data_pos = tf.random.truncated_normal(shape=[800*self.batch_size, 2],mean = 0, stddev = 0.1)
		neg = tf.random.uniform(shape = [800*self.batch_size, 2], minval = [0.2,0], maxval = [0.5,2*np.pi]).numpy()
		data_neg = np.concatenate((np.expand_dims(neg[:,0]*np.cos(neg[:,1]),axis=1),np.expand_dims(neg[:,0]*np.sin(neg[:,1]), axis = 1)), axis = 1)
		# neg = tf.random.truncated_normal(shape=[200*self.batch_size, 2],mean = 0, stddev = 0.5)

		print(data_neg,data_neg.shape)
		# print(X)
		# data_neg = data_neg[][0:800]
		return data_pos, data_neg

	def dataset_g2(self, train_data_pos, train_data_neg, batch_size):
		train_dataset_pos = tf.data.Dataset.from_tensor_slices((train_data_pos))
		train_dataset_pos = train_dataset_pos.shuffle(50000)
		train_dataset_pos = train_dataset_pos.batch(batch_size)
		train_dataset_pos = train_dataset_pos.prefetch(5)

		train_dataset_neg = tf.data.Dataset.from_tensor_slices((train_data_neg))
		train_dataset_neg = train_dataset_neg.shuffle(50000)
		train_dataset_neg = train_dataset_neg.batch(batch_size)
		train_dataset_neg = train_dataset_neg.prefetch(5)

		return train_dataset_pos, train_dataset_neg 

	def gen_func_gmm8_NEEDFIX(self):
		self.MIN = -4
		self.MAX = 4
		tfd = tfp.distributions
		probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
		locs = [[2., 0.], [0., 2.], [-2.,0.], [0.,-2.], [2*0.7071, 2*0.7071], [-2*0.7071, 2*0.7071], [2*0.7071, -2*0.7071], [-2*0.7071, -2*0.7071] ]
		# covs = [ [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]]   ]

		gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=[.0004, .0004, .0004, .0004, .0004, .0004, .0004, .0004]))

		return gmm.sample(sample_shape=(int(1000*self.batch_size.numpy())))

	def dataset_gmm8_NEEDFIX(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset



	def gen_func_comma(self):
		import h5py
		try:
			self.h5 = h5py.File('data/CommaAI/CommaAI_dataset_RumiGAN.h5', 'r')
		except:
			try:
				with open("data/CommaAI/Comma_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
				with open("data/CommaAI/Comma_Logs.txt","r") as names:
					log_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('data/CommaAI/camera/*.h5'))
				log_files = sorted(glob.glob('data/CommaAI/log/*.h5'))
				print("Data File Created. Saving filenames")
				with open("data/CommaAI/Comma_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
				with open("data/CommaAI/Comma_Logs.txt","w") as logs:
					for log in log_files:
						logs.write(str(log)+'\n')
			
			'''
			Code Courtesy Comma AI research: 
			https://github.com/commaai/research/blob/master/dask_generator.py 
			'''
			angle = []  # steering angle of the car
			speed = []  # steering angle of the car
			hdf5_camera = []  # the camera hdf5 files need to continue open
			c5x = []
			filters = []
			x_vec = []
			lastidx = 0
			time_len = 1
			#Need to read out data and steeding angles
			print(true_files)
			print(log_files)
			for cword, tword in zip(true_files[0:5], log_files[0:5]):
				print(cword,tword)
				try:
					with h5py.File(tword, "r") as t5:
						c5 = h5py.File(cword, "r")
						hdf5_camera.append(c5)
						x = c5["X"]
						c5x.append((lastidx, lastidx+x.shape[0], x))


						speed_value = t5["speed"][:]
						steering_angle = t5["steering_angle"][:]
						idxs = np.linspace(0, steering_angle.shape[0]-1, x.shape[0]).astype("int")  # approximate alignment
						xs = x[:]
						if lastidx == 0:
							x_vec = np.array(xs)
						else:
							x_vec = np.concatenate([x_vec, xs], axis = 0)
						# x_vec.append(xs)
						angle.append(steering_angle[idxs])
						speed.append(speed_value[idxs])

						goods = np.abs(angle[-1]) <= 200

						filters.append(np.argwhere(goods)[time_len-1:] + (lastidx+time_len-1))
						lastidx += goods.shape[0]
						# check for mismatched length bug
						print("x {} | t {} | f {} | xv {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0], xs.shape[0]))
						if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
							raise Exception("bad shape")
						print(x.shape, steering_angle.shape, goods.shape)

				except IOError:
					import traceback
					traceback.print_exc()
					print('failed to open'), tword

			
			# self.h5 = h5py.File('data/CommaAI/CommaAI_dataset.h5', 'r')
			# f = np.array(self.h5['frames'])
			# a = np.array(self.h5['angle'])
			# p = f[np.where(a > 150)[0]]
			# n = f[np.where(a <= 150)[0]]


			# angle = np.concatenate(angle, axis=0)
			# speed = np.concatenate(speed, axis=0)
			# filters = np.concatenate(filters, axis=0).ravel()

			# true_frames = x_vec[filters]
			# true_angle = angle[filters]
			# true_speed = speed[filters]

			# hf = h5py.File('CommaAI_dataset_RumiGAN.h5', 'w')
			# hf.create_dataset('frames_pos', data=p)
			# hf.create_dataset('frames_neg', data=n)
			# hf.create_dataset('angle', data=a)
			# # hf.create_dataset('speed', data=true_speed)
			# hf.close()

			self.h5 = h5py.File('data/CommaAI/CommaAI_dataset_RumiGAN.h5', 'r')

			print('training on {}/{} examples'.format(frames_pos.shape[0], angle.shape[0]))
			print(angle, filters, speed)
			print(angle.shape, filters.shape, speed.shape)

			print('training on {} pos examples, {} angles'.format(p.shape[0], true_angle.shape[0]))
			print('training on {} neg examples, {} angles'.format(n.shape[0], true_angle.shape[0]))
		
		#### Create a sequences for randomly selecting batch from the h5. 

		self.h5_pos = h5py.File('data/CommaAI/CommaAI_dataset_RumiGAN.h5', 'r')
		# a list of number of batches possible
		print('training on {}/{} pos examples'.format(self.h5_pos['frames'].shape[0], self.h5_pos['angle'].shape[0]))
		print('training on {}/{} neg examples'.format(self.h5['frames_neg'].shape[0], self.h5['angle'].shape[0]))

		self.ratio = self.h5['frames_pos'].shape[0] / float(self.h5['frames_neg'].shape[0])



		print(self.ratio)
		self.p_sel_vec = np.arange(self.h5_pos['frames'].shape[0])
		print(self.p_sel_vec)
		np.random.shuffle(self.p_sel_vec)
		print(self.p_sel_vec)

		self.n_sel_vec = np.arange(self.h5['frames_neg'].shape[0])
		print(self.n_sel_vec)
		np.random.shuffle(self.n_sel_vec)
		print(self.n_sel_vec)


		self.num_h5_batches = int(max(self.h5_pos['frames'].shape[0],self.h5['frames_neg'].shape[0]) / self.batch_size)

		self.p_sel_vec = np.tile(self.p_sel_vec,int(np.ceil(1/self.ratio)))
		self.n_sel_vec = np.tile(self.n_sel_vec,int(np.ceil(self.ratio)))

		print(self.p_sel_vec.shape,self.n_sel_vec.shape)
		minshape = min(self.p_sel_vec.shape[0],self.n_sel_vec.shape[0])
		self.p_sel_vec = self.p_sel_vec[:minshape]
		self.n_sel_vec = self.n_sel_vec[:minshape]
		print(self.p_sel_vec.shape,self.n_sel_vec.shape)


		# print(self.sel_vec)
		# print(self.h5['frames'][self.sel_vec])
		return self.p_sel_vec, self.n_sel_vec

	def dataset_comma(self, train_data_pos, train_data_neg, batch_size):
		class generator_p:
			def __init__(self,h5,sel_vec):
				self.h5_var = h5
				self.sel_v = sel_vec
				np.random.shuffle(self.sel_v)

			def __call__(self):
				for sel in self.sel_v:
					yield self.h5_var['frames'][sel]

		class generator_n:
			def __init__(self,h5,sel_vec):
				self.h5_var = h5
				self.sel_v = sel_vec
				np.random.shuffle(self.sel_v)

			def __call__(self):
				for sel in self.sel_v:
					yield self.h5_var['frames_neg'][sel]

		def resize_func(im):
			im = tf.transpose(im, [1, 2, 0])
			im = tf.image.resize(im, [self.output_H, self.output_W])
			im = tf.divide(im, tf.constant(255, dtype = 'float32'))
			return im

		train_dataset_pos = tf.data.Dataset.from_generator(generator_p(self.h5_pos, self.p_sel_vec), (tf.uint8), (tf.TensorShape([ 3, 160, 320]))) 
		train_dataset_pos = train_dataset_pos.map(resize_func, num_parallel_calls = int(self.num_parallel_calls))
		train_dataset_pos = train_dataset_pos.shuffle(4)
		train_dataset_pos = train_dataset_pos.batch(batch_size)
		train_dataset_pos = train_dataset_pos.prefetch(5)

		train_dataset_neg = tf.data.Dataset.from_generator(generator_n(self.h5, self.n_sel_vec), (tf.uint8), (tf.TensorShape([ 3, 160, 320]))) 
		train_dataset_neg = train_dataset_neg.map(resize_func, num_parallel_calls = int(self.num_parallel_calls))
		train_dataset_neg = train_dataset_neg.shuffle(4)
		train_dataset_neg = train_dataset_neg.batch(batch_size)
		train_dataset_neg = train_dataset_neg.prefetch(5)
		# train_dataset = train_dataset.cache(filename=self.cache_loc)
		return train_dataset_pos,train_dataset_neg


'''***********************************************************************************
********** GAN_DATA_ACGAN ************************************************************
***********************************************************************************'''
class GAN_DATA_CondGAN(GAN_DATA_ops):

	def __init__(self):
		GAN_DATA_ops.__init__(self)

	def gen_func_mnist(self):
		self.num_classes = 10
		train_images, train_labels, test_images, test_labels = self.mnist_loader()

		zero = train_labels == 0
		one = train_labels == 1
		two  = train_labels == 2
		three  = train_labels == 3
		four  = train_labels == 4
		five  = train_labels == 5
		six  = train_labels == 6
		seven  = train_labels == 7
		eight = train_labels == 8
		nine = train_labels == 9
		self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
		self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
		self.fid_images_single = train_images[np.where(train_labels == self.number)[0]]
		self.fid_images_sharp = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]

		if self.testcase == 'single':
			self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_images_single = train_images[np.where(train_labels == self.number)[0]]
			self.fid_images_sharp = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			# train_images = train_images[np.where(train_labels == self.number)[0]]
			# train_labels = train_labels[np.where(train_labels == self.number)[0]]
			# self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
		if self.testcase == 'few':
			self.fid_images_single = train_images[np.where(train_labels == self.number)[0]][0:self.num_few]
			self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
			
			self.fid_images_sharp = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]

			# train_images = np.concatenate( (np.repeat(train_images[np.where(train_labels == self.number)[0]][0:100],50,axis = 0), train_images[np.where(train_labels != self.number)[0]]), axis = 0)
			# train_labels = np.concatenate( (np.repeat(train_labels[np.where(train_labels == self.number)[0]][0:100],50,axis = 0), train_labels[np.where(train_labels != self.number)[0]]), axis = 0)

			##num_few typically 200

			train_images = np.concatenate( (train_images[np.where(train_labels == self.number)[0]][0:self.num_few], train_images[np.where(train_labels != self.number)[0]]), axis = 0)
			train_labels = np.concatenate( (train_labels[np.where(train_labels == self.number)[0]][0:self.num_few], train_labels[np.where(train_labels != self.number)[0]]), axis = 0)

			# self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
			# self.fid_train_images_few = train_images

			
			zero = train_labels == 0
			one = train_labels == 1
			two  = train_labels == 2
			three  = train_labels == 3
			four  = train_labels == 4
			five  = train_labels == 5
			six  = train_labels == 6
			seven  = train_labels == 7
			eight = train_labels == 8
			nine = train_labels == 9

			#train_images[np.where(train_labels == self.number)[0][0:500]]
		if self.testcase == 'even':
			# train_images = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_images_single = train_images[np.where(train_labels == self.number)[0]]
			self.fid_images_sharp = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]

		if self.testcase == 'odd':
			# train_images = train_images[np.where(train_labels%2 != 0)[0]]
			# self.fid_train_images = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_images_single = train_images[np.where(train_labels == self.number)[0]]
			self.fid_images_sharp = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
		if self.testcase == 'sharp':
			self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_images_single = train_images[np.where(train_labels == self.number)[0]]
			self.fid_images_sharp = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			# train_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			# self.fid_train_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]

		self.reps = int(60000.0/train_images.shape[0])
		# self.target_fake_output = tf.one_hot(list(10*np.ones([self.batch_size, 1])),depth = 11)

		return train_images, train_labels

	def dataset_mnist(self,train_data,train_labels,batch_size):

		# if self.label_style =='base':
		# 	train_labels = tf.one_hot(np.squeeze(train_labels),depth = self.num_classes)
		# print(train_labels)

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
		if self.testcase == 'single':
			train_dataset = train_dataset.repeat(self.reps-1)
		train_dataset = train_dataset.shuffle(60000)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(10)
		return train_dataset

	def gen_func_celeba(self):

		self.num_classes = 2
		train_images, data_array = self.celeba_loader()
		# print(data_array,data_array.shape)
		tags = data_array[0,:] # print to find which col to pull for what
		gender = data_array[1:,21]
		bald_tag = data_array[1:,5]
		hat_labels = data_array[1:,-5]
		# print(gender,gender.shape)

		male = gender == '1'
		male = male.astype('uint8')

		bald = bald_tag == '1'
		bald = bald.astype('uint8')

		hat = hat_labels == '1'
		hat = hat.astype('uint8')

		# train_labels = np.expand_dims(np.array(male), 1).astype('float32')

		#num_few typicaly 10k

		if self.testcase == 'fewfemale':
			train_labels = np.expand_dims(np.array(male), 1).astype('float32')
			self.fid_train_images = train_images[np.where(male == 1)]
			train_images = np.concatenate( (np.repeat(train_images[np.where(male == 0)][0:self.num_few],20,axis = 0), train_images[np.where(male == 1)]), axis = 0)
			train_labels = np.concatenate( (np.repeat(train_labels[np.where(male == 0)][0:self.num_few],20,axis = 0), train_labels[np.where(male == 1)]), axis = 0)
		if self.testcase == 'fewmale':
			train_labels = np.expand_dims(np.array(male), 1).astype('float32')
			self.fid_train_images = train_images[np.where(male == 0)]
			train_images = np.concatenate( (np.repeat(train_images[np.where(male == 1)][0:self.num_few],20,axis = 0), train_images[np.where(male == 0)]), axis = 0)
			train_labels = np.concatenate( (np.repeat(train_labels[np.where(male == 1)][0:self.num_few],20,axis = 0), train_labels[np.where(male == 0)]), axis = 0)
		if self.testcase == 'female':
			train_labels = np.expand_dims(np.array(male), 1).astype('float32')
			self.fid_train_images = train_images[np.where(male == 0)]
		if self.testcase == 'male':
			train_labels = np.expand_dims(np.array(male), 1).astype('float32')
			self.fid_train_images = train_images[np.where(male == 1)]
		if self.testcase == 'bald':
			train_labels = np.expand_dims(np.array(bald), 1).astype('float32')
			self.fid_train_images = train_images[np.where(bald == 1)]
		if self.testcase == 'hat':
			train_labels = np.expand_dims(np.array(hat), 1).astype('float32')
			self.fid_train_images = train_images[np.where(hat == 1)]


		return train_images, train_labels

	def dataset_celeba(self,train_data,train_labels,batch_size):	
		def data_reader_faces(filename,label):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[self.output_size,self.output_size])
				# This will convert to float values in [-1, 1]
				image = tf.divide(image,255.0)
				# image = tf.subtract(image,127.0)
				# image = tf.divide(image,127.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image,label

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data,train_labels))
		train_dataset = train_dataset.shuffle(200000)
		train_dataset = train_dataset.map(data_reader_faces, num_parallel_calls=int(self.num_parallel_calls))
		train_dataset = train_dataset.batch(batch_size, drop_remainder = True)
		train_dataset = train_dataset.prefetch(15)
		return train_dataset

	def gen_func_cifar10(self):
		self.num_classes = 10
		train_images, train_labels, test_images, test_labels = self.cifar10_loader()

		zero = train_labels == 0
		one = train_labels == 1
		two  = train_labels == 2
		three  = train_labels == 3
		four  = train_labels == 4
		five  = train_labels == 5
		six  = train_labels == 6
		seven  = train_labels == 7
		eight = train_labels == 8
		nine = train_labels == 9

		if self.testcase == 'single':	
			self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
		if self.testcase == 'few':	
			self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
			# train_images = np.concatenate( (np.repeat(train_images[np.where(train_labels == self.number)[0]][0:self.num_few],5,axis = 0), train_images[np.where(train_labels != self.number)[0]]), axis = 0)
			# train_labels = np.concatenate( (np.repeat(train_labels[np.where(train_labels == self.number)[0]][0:self.num_few],5,axis = 0), train_labels[np.where(train_labels != self.number)[0]]), axis = 0)
			train_images = np.concatenate( (train_images[np.where(train_labels == self.number)[0]][0:self.num_few], train_images[np.where(train_labels != self.number)[0]]), axis = 0)
			train_labels = np.concatenate( (train_labels[np.where(train_labels == self.number)[0]][0:self.num_few], train_labels[np.where(train_labels != self.number)[0]]), axis = 0)
		if self.testcase == 'even':
			self.fid_train_images = train_images[np.where(train_labels%2 == 0)[0]]
		if self.testcase == 'odd':
			self.fid_train_images = train_images[np.where(train_labels%2 != 0)[0]]
		if self.testcase == 'sharp':
			self.fid_train_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
		if self.testcase == 'animals':
			self.fid_train_images = train_images[np.where(np.any([two, three, four, five, six, seven],axis=0))[0]]

		self.reps = int(60000.0/train_images.shape[0])
		# self.target_fake_output = tf.one_hot(list(10*np.ones([self.batch_size, 1])),depth = 11)
		print(train_images.shape, train_labels.shape)

		return train_images, train_labels

	def dataset_cifar10(self,train_data,train_labels,batch_size):
		
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data,train_labels))
		if self.testcase == 'single':
			train_dataset = train_dataset.repeat(self.reps-1)
		train_dataset = train_dataset.shuffle(50000)
		train_dataset = train_dataset.batch(batch_size,drop_remainder = True)
		train_dataset = train_dataset.prefetch(10)
		return train_dataset

	def gen_func_g1(self):
		# self.MIN = -1
		# self.MAX = 12
		g1 = tfp.distributions.TruncatedNormal(loc=0.2, scale=0.1, low=0., high=1.)
		return g1.sample([1000*self.batch_size, 1])
		# return tf.random.normal([1000*self.batch_size, 1], mean = 8.0, stddev = 1.)


	def dataset_g1(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset

	def gen_func_gN(self):
		g1 = tfp.distributions.TruncatedNormal(loc=0.75, scale=0.1, low=0., high=1.)
		return g1.sample([100*self.batch_size, self.output_size])
		# return tf.random.normal([20*self.batch_size, self.GaussN], mean = 7.0*np.ones((1,self.output_size)), stddev = 1.*np.ones((1,self.output_size)))

	def dataset_gN(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(1000)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset

	def gen_func_gmm2(self):
		self.MIN = -1
		self.MAX = 12
		tfd = tfp.distributions
		probs = [0.5, 0.25, 0.25]
		locs = [[1.2],[6.0], [8.7]]
		# stddev_scale = [.04, .04, .04, .04, .04, .04, .04, .04]
		stddev_scale = [0.21, 0.2, 0.21]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# covs = [ [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]]   ]

		gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		return gmm.sample(sample_shape=(int(1000*self.batch_size.numpy())))

	def dataset_gmm2(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(10)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(10)
		return train_dataset

	def gen_func_g2(self):
		self.MIN = -0.2
		self.MAX = 1.2
		return tf.random.normal([1000*self.batch_size.numpy(), 2], mean =np.array([0.70,0.70]), stddev = np.array([0.05,0.05]))

	def dataset_g2(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset

	def gen_func_gmm8(self):
		self.MIN = -0.2
		self.MAX = 1.2
		tfd = tfp.distributions
		probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
		# locs = [[2., 0.], [0., 2.], [-1.,0.], [0.,-1.], [1*0.7071, 1*0.7071], [-1*0.7071, 1*0.7071], [1*0.7071, -1*0.7071], [-1*0.7071, -1*0.7071] ]
		# locs = [[0.25, 0.], [0., 0.25], [-0.25,0.], [0.,-0.25], [0.25*0.7071, 0.5*0.7071], [-0.25*0.7071, 0.25*0.7071], [0.25*0.7071, -0.25*0.7071], [-0.25*0.7071, -0.25*0.7071] ]
		# locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.75*0.7071, 0.75*0.7071], [0.25*0.7071, 0.75*0.7071], [0.75*0.7071, 0.25*0.7071], [0.25*0.7071, 0.25*0.7071] ]
		locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.5*1.7071, 0.5*1.7071], [0.5*0.2929, 0.5*1.7071], [0.5*1.7071, 0.5*0.2929], [0.5*0.2929, 0.5*0.2929] ]

		# stddev_scale = [.04, .04, .04, .04, .04, .04, .04, .04]
		stddev_scale = [.03, .03, .03, .03, .03, .03, .03, .03]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# covs = [ [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]]   ]

		gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		return gmm.sample(sample_shape=(int(100*self.batch_size.numpy())))

	def dataset_gmm8(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset











