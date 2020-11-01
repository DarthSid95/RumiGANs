from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json
import glob
from tqdm.autonotebook import tqdm
import shutil
import warnings

# import tensorflow_probability as tfp
# tfd = tfp.distributions

##FOR FID
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
import scipy as sp
from numpy import iscomplexobj


from arch import *
from gan_metrics import *

'''
GAN_SRC Consists of the common parts of GAN architectures, speficially, the calls to the sub architecture classes from the respective files, and the calls for FID evaluations. Each ARCH_data_* class has archtectures corresponding to that dataset learning and for the loss case ( Autoencoder structures for DEQ case, etc.)
'''
'''***********************************************************************************
********** GAN Source Class -- All the basics and metrics ****************************
***********************************************************************************'''
class GAN_SRC(eval('ARCH_'+FLAGS.data), GAN_Metrics):

	def __init__(self,FLAGS_dict):
		''' Defines anything common to te diofferent GAN approaches. Architectures of Gen and Disc, all flags,'''
		for name,val in FLAGS_dict.items():
			exec('self.'+name+' = val')

		if self.colab:
			if self.pbar_flag:
				warnings.warn("Repeated updation of the tqdm progress bar on Colab can cause OOM on Colab, resulting in pkill, or OOM on the local system, causing the browser to hang.",ResourceWarning)
			if self.latex_plot_flag:
				warnings.warn("Plotting latex on colab require insalling the texlive library in your colab instance. Not doing so will case errors while plotting.",ImportWarning)


		# if self.colab and (self.data in ['mnist', 'celeba', 'cifar10']):
		# 	self.bar_flag = 0
		# else:
		# 	self.bar_flag = 1


		if self.device == '-1':
			self.device = '/CPU'
		elif self.device == '':
			self.device = '/CPU'
		else:
			self.device = '/GPU:'+self.device
			
		print(self.device)

		with tf.device(self.device):
			self.batch_size = tf.constant(self.batch_size,dtype='int64')
			self.fid_batch_size = tf.constant(100,dtype='int64')
			self.num_epochs = tf.constant(self.num_epochs,dtype='int64')
			self.Dloop = tf.constant(self.Dloop,dtype='int64')
			self.Gloop = tf.constant(self.Gloop,dtype='int64')
			self.lr_D = tf.constant(self.lr_D)
			self.lr_G = tf.constant(self.lr_G)
			self.beta1 = tf.constant(self.beta1)
			self.total_count = tf.Variable(0,dtype='int64')


		eval('ARCH_'+self.data+'.__init__(self)')

		self.num_to_print = 10

		if self.mode in ['test','metrics']:
			self.num_test_images = 20
		else:
			self.num_test_images = 10


		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.3e}', 2: f'{0:2.3e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining} Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'


		if self.log_folder == 'default':
			today = date.today()
			self.log_dir = 'logs/Log_Folder_'+today.strftime("%d%m%Y")+'/'
		else:
			self.log_dir = self.log_folder
		
		if self.log_dir[-1] != '/':
			self.log_dir += '/'	

		self.run_id_flag = self.run_id
		
		self.create_run_location()

		self.timestr = time.strftime("%Y%m%d-%H%M%S")
		if self.res_flag == 1:
			self.res_file = open(self.run_loc+'/'+self.run_id+'_Results.txt','a')
			FLAGS.append_flags_into_file(self.run_loc+'/'+self.run_id+'_Flags.txt')


		GAN_Metrics.__init__(self)


	def create_run_location(self):
		''' If resuming, locate the file to resule and set the current running direcrtory. Else, create one based on the data cases given.'''

		''' Create log folder / Check for existing log folder'''
		if os.path.exists(self.log_dir):
			print("Directory " , self.log_dir ,  " already exists")
		else:
			os.mkdir(self.log_dir)
			print("Directory " , self.log_dir ,  " Created ")   

		if self.resume:		
			self.run_loc = self.log_dir + self.run_id
			print("Resuming from folder {}".format(self.run_loc))
		else:
			print("No RunID specified. Logs will be saved in a folder based on FLAGS")	
			today = date.today()
			d1 = today.strftime("%d%m%Y")
			self.run_id = d1 +'_'+ self.topic + '_' + self.data + '_' + self.gan + '_' + self.loss
			self.run_loc = self.log_dir + self.run_id

			runs = sorted(glob.glob(self.run_loc+'*/'))
			print(runs)
			if len(runs) == 0:
				curnum = 0
			else:
				curnum = int(runs[-1].split('_')[-1].split('/')[0])
			print(curnum)
			if self.run_id_flag == 'new':
				self.curnum = curnum+1
			else:
				self.curnum = curnum
				if self.run_id_flag != 'same' and os.path.exists(self.run_loc + '_' + str(self.curnum).zfill(2)):
					x = input("You will be OVERWRITING existing DATA. ENTER to continue, type N to create new ")
					if x in ['N','n']:
						self.curnum += 1
			self.run_loc += '_'+str(self.curnum).zfill(2)



		if os.path.exists(self.run_loc):
			print("Directory " , self.run_loc ,  " already exists")
		else:   
			if self.resume:
				print("Cannot resume. Specified log does not exist")
			else:	
				os.mkdir(self.run_loc)
				print("Directory " , self.run_loc ,  " Created ") 



		self.checkpoint_dir = self.run_loc+'/checkpoints'
		if os.path.exists(self.checkpoint_dir):
			print("Checkpoint directory " , self.checkpoint_dir ,  " already exists")
		else:
			os.mkdir(self.checkpoint_dir)
			print("Checkpoint directory " , self.checkpoint_dir ,  " Created ")  



		self.im_dir = self.run_loc+'/Images'
		if os.path.exists(self.im_dir):
			print("Images directory " , self.im_dir ,  " already exists")
		else:
			os.mkdir(self.im_dir)
			print("Images directory " , self.im_dir ,  " Created ") 
		self.impath = self.im_dir + '/Images_'



		self.metric_dir = self.run_loc+'/Metrics'
		if os.path.exists(self.metric_dir):
			print("Metrics directory " , self.metric_dir ,  " already exists")
		else:
			os.mkdir(self.metric_dir)
			print("Metrics directory " , self.metric_dir ,  " Created ")
		self.metricpath = self.metric_dir + '/Metrics_'

			


	def get_terminal_width(self):
		width = shutil.get_terminal_size(fallback=(200, 24))[0]
		if width == 0:
			width = 120
		return width


	def pbar(self, epoch):
		bar = tqdm(total=(int(self.train_dataset_size*self.reps) // int(self.batch_size.numpy())) * int(self.batch_size.numpy()), ncols=int(self.get_terminal_width() * .9), desc=tqdm.write(f' \n Epoch {int(epoch)}/{int(self.num_epochs.numpy())}'), postfix=self.postfix, bar_format=self.bar_format, unit = ' Samples')
		return bar


	def generate_and_save_batch(self,epoch = 999):
		
		path = self.impath + str(self.total_count.numpy())
		label = 'Epoch {0}'.format(epoch)

		noise = tf.random.normal([self.num_to_print*self.num_to_print, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)

		if self.topic in ['cGAN', 'ACGAN']:
			class_vec = []
			for i in range(self.num_classes):
				class_vec.append(i*np.ones(int((self.num_to_print**2)/self.num_classes)))
			class_final = np.expand_dims(np.concatenate(class_vec,axis = 0),axis = 1)
			if self.label_style == 'base':
				class_final = tf.one_hot(np.squeeze(class_final), depth = self.num_classes)
			predictions = self.generator([noise,class_final], training=False)
		else:
			predictions = self.generator(noise, training=False)

		if self.data != 'celeba':
			predictions = (predictions + 1.0)/2.0

		self.save_image_batch(images = predictions,label = label, path = path)


	def save_image_batch(self, images = None, label = 'Default Image Label', path = 'result.png'):

		images_on_grid = self.image_grid(input_tensor = images, grid_shape = (self.num_to_print,self.num_to_print), image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		fig1 = plt.figure(figsize=(14,14))
		ax1 = fig1.add_subplot(111)
		ax1.cla()
		ax1.axis("off")
		if images_on_grid.shape[2] == 3:
			ax1.imshow(np.clip(images_on_grid,0.,1.))
		else:
			ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')
		plt.title(label, fontsize=12)
		plt.tight_layout()
		plt.savefig(path)
		plt.close()


	def image_grid(self,input_tensor, grid_shape, image_shape=(32, 32), num_channels=3):
		"""Arrange a minibatch of images into a grid to form a single image.
		Args:
		input_tensor: Tensor. Minibatch of images to format, either 4D
			([batch size, height, width, num_channels]) or flattened
			([batch size, height * width * num_channels]).
		grid_shape: Sequence of int. The shape of the image grid,
			formatted as [grid_height, grid_width].
		image_shape: Sequence of int. The shape of a single image,
			formatted as [image_height, image_width].
		num_channels: int. The number of channels in an image.
		Returns:
		Tensor representing a single image in which the input images have been
		arranged into a grid.
		Raises:
		ValueError: The grid shape and minibatch size don't match, or the image
			shape and number of channels are incompatible with the input tensor.
		"""
		num_padding = int(np.ceil(0.02*image_shape[0]))
		paddings = tf.constant([[0, 0], [num_padding, num_padding], [num_padding, num_padding], [0, 0]])
		image_shape = (image_shape[0]+(2*num_padding), image_shape[1]+(2*num_padding))
		input_tensor = tf.pad(input_tensor, paddings, "CONSTANT", constant_values = 1.0)

		if grid_shape[0] * grid_shape[1] != int(input_tensor.shape[0]):
			raise ValueError("Grid shape %s incompatible with minibatch size %i." %
						 (grid_shape, int(input_tensor.shape[0])))
		if len(input_tensor.shape) == 2:
			num_features = image_shape[0] * image_shape[1] * num_channels
			if int(input_tensor.shape[1]) != num_features:
				raise ValueError("Image shape and number of channels incompatible with "
							   "input tensor.")
		elif len(input_tensor.shape) == 4:
			if (int(input_tensor.shape[1]) != image_shape[0] or \
				int(input_tensor.shape[2]) != image_shape[1] or \
				int(input_tensor.shape[3]) != num_channels):
				raise ValueError("Image shape and number of channels incompatible with input tensor. %s vs %s" % (input_tensor.shape, (image_shape[0], image_shape[1],num_channels)))
		else:
			raise ValueError("Unrecognized input tensor format.")

		height, width = grid_shape[0] * image_shape[0], grid_shape[1] * image_shape[1]
		input_tensor = tf.reshape(input_tensor, tuple(grid_shape) + tuple(image_shape) + (num_channels,))
		input_tensor = tf.transpose(a = input_tensor, perm = [0, 1, 3, 2, 4])
		input_tensor = tf.reshape(input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
		input_tensor = tf.transpose(a = input_tensor, perm = [0, 2, 1, 3])
		input_tensor = tf.reshape(input_tensor, [1, height, width, num_channels])
		return input_tensor[0]


	def h5_from_checkpoint(self):
		self.generate_and_save_batch()
		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)




