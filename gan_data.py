from __future__ import print_function
import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
# tfd = tfp.distributions
import glob
from absl import flags
import csv

from scipy import io as sio

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import PdfPages

# import tensorflow_datasets as tfds
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
		self.noise_dims = 100
		if self.data == 'celeba':
			self.output_size = self.out_size
		elif self.data == 'mnist':
			self.output_size = 28
		elif self.data == 'cifar10':
			self.output_size = 32

		# self.testcase = testcase 
		# self.number = number

	def mnist_loader(self):
		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
		# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
		train_images = train_images.reshape(train_images.shape[0],self.output_size, self.output_size, 1).astype('float32')
		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		test_images = test_images.reshape(test_images.shape[0],self.output_size, self.output_size, 1).astype('float32')
		self.test_images = (test_images - 127.5) / 127.5

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
		print(train_images)

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
		if self.testcase == 'even':
			train_images = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'odd':
			train_images = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_train_images = train_images
		if self.testcase == 'overlap':
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
		if self.testcase == 'overlap':
			true_images = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			false_images = train_images[np.where(np.any([zero, two, three, six, eight, nine],axis=0))[0]]
			self.fid_train_images = true_images
			self.fid_others = false_images
		if self.testcase == 'overlapSVHN':
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

	def dataset_celeba(self, train_data_pos, train_data_neg, batch_size):
		
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
		self.fid_images_overlap = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]

		if self.testcase == 'single':
			self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_images_single = train_images[np.where(train_labels == self.number)[0]]
			self.fid_images_overlap = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
			# train_images = train_images[np.where(train_labels == self.number)[0]]
			# train_labels = train_labels[np.where(train_labels == self.number)[0]]
			# self.fid_train_images = train_images[np.where(train_labels == self.number)[0]]
		if self.testcase == 'few':
			self.fid_images_single = train_images[np.where(train_labels == self.number)[0]][0:self.num_few]
			self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
			
			self.fid_images_overlap = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]

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
			self.fid_images_overlap = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]

		if self.testcase == 'odd':
			# train_images = train_images[np.where(train_labels%2 != 0)[0]]
			# self.fid_train_images = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_images_single = train_images[np.where(train_labels == self.number)[0]]
			self.fid_images_overlap = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
		if self.testcase == 'overlap':
			self.fid_images_even = train_images[np.where(train_labels%2 == 0)[0]]
			self.fid_images_odd = train_images[np.where(train_labels%2 != 0)[0]]
			self.fid_images_single = train_images[np.where(train_labels == self.number)[0]]
			self.fid_images_overlap = train_images[np.where(np.any([one, two, four, five, seven, nine],axis=0))[0]]
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
		if self.testcase == 'overlap':
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











