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

import tensorflow_probability as tfp
tfd = tfp.distributions

##FOR FID
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
import scipy as sp
from numpy import iscomplexobj


from arch import *
from ops import *

'''
GAN_SRC Consists of the common parts of GAN architectures, speficially, the calls to the sub architecture classes from the respective files, and the calls for FID evaluations. Each ARCH_data_* class has archtectures corresponding to that dataset learning and for the loss case ( Autoencoder structures for DEQ case, etc.)
'''
'''***********************************************************************************
********** GAN Source Class -- All the basics and metrics ****************************
***********************************************************************************'''
class GAN_SRC(eval('ARCH_'+FLAGS.data)): #mnist, ARCH_celeba, ARCG_g1, ARCH_g2, ARCH_gmm8, ARCH_comma):  eval('ARCH_'+FLAGS.data),

	def __init__(self,FLAGS_dict):
		''' Defines anything common to te diofferent GAN approaches. Architectures of Gen and Disc, all flags,'''
		for name,val in FLAGS_dict.items():
			exec('self.'+name+' = val')


		if self.colab and (self.data in ['mnist', 'celeba', 'cifar10']):
			self.bar_flag = 0
		else:
			self.bar_flag = 1


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
		self.setup_metrics()


		self.timestr = time.strftime("%Y%m%d-%H%M%S")
		if self.res_flag == 1:
			self.res_file = open(self.run_loc+'/'+self.run_id+'_Results.txt','a')
			FLAGS.append_flags_into_file(self.run_loc+'/'+self.run_id+'_Flags.txt')


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
		if self.loss == 'FS' and self.topic != 'AAE':
			self.impath += self.latent_kind+'_'



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


	def generate_and_save_batch(self,epoch):
		noise = tf.random.normal([self.num_to_print*self.num_to_print, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)
		path = self.impath + str(self.total_count.numpy())

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
		eval(self.show_result_func)

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

	def model_saver(self):
		self.generate_and_save_batch(999)
		self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
		self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

	def setup_metrics(self):
		self.FID_flag = 0
		self.PR_flag = 0
		self.class_prob_flag = 0
		self.metric_counter_vec = []


		if 'FID' in self.metrics:
			self.FID_flag = 1
			self.FID_load_flag = 0
			self.FID_vec = []
			self.FID_vec_new = []

			if self.data in ['mnist']:
				self.FID_steps = 500
				if self.mode == 'metrics':
					self.FID_num_samples = 10000
				else:
					self.FID_num_samples = 5000#15000
			elif self.data in ['cifar10']:
				self.FID_steps = 500
				if self.mode == 'metrics':
					self.FID_num_samples = 10000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['celeba']:
				self.FID_steps = 2500
				if self.mode == 'metrics':
					self.FID_num_samples = 10000
				else:
					self.FID_num_samples = 5000
			else:
				self.FID_flag = 0
				print('FID cannot be evaluated on this dataset')


		if 'PR' in self.metrics:
			### NEed to DeisGN
			self.PR_flag = 1
			self.PR_vec = []
			self.PR_steps = self.FID_steps


		if 'ClassProbs' in self.metrics:
			self.class_prob_vec = []

			if self.data in 'mnist':
				self.class_prob_flag = 1
				self.class_prob_steps = 100 
				self.classifier_load_flag = 0
			else:
				print("Cannot find class-wise probabilites for this dataset")

	def eval_metrics(self):
		update_flag = 0

		if self.FID_flag and (self.total_count.numpy()%self.FID_steps == 0 or self.mode == 'metrics'):
			update_flag = 1
			self.update_FID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'FID.npy',np.array(self.FID_vec))
				if self.topic == 'RumiGAN' and self.data == 'mnist':
					self.print_FID_Rumi()
				elif self.topic in['cGAN', 'ACGAN'] and self.data == 'mnist':
					self.print_FID_ACGAN()
				else:
					self.print_FID()


		if self.PR_flag and (self.total_count.numpy()%self.PR_steps == 0 or self.mode == 'metrics'):
			update_flag = 1
			self.update_PR()
			if self.mode != 'metrics':
				np.save(self.metricpath+'PR_all.npy',np.array(self.PR_vec))
				self.print_PR()
			else:
				np.save(self.metricpath+'PR_MetricsEval.npy',np.array(self.PR_vec))
				self.print_PR()


		if self.class_prob_flag and (self.total_count.numpy()%self.class_prob_steps == 0 or self.mode == 'metrics'):
			update_flag = 1
			self.class_prob_metric()
			if self.mode != 'metrics':
				np.save(self.metricpath+'ClassProbs.npy',np.array(self.class_prob_vec))
				self.print_ClassProbs()

		if self.res_flag and update_flag:
			self.res_file.write("Metrics avaluated at Iteration " + str(self.total_count.numpy()) + '\n')


	def update_PR(self):
		self.PR = compute_prd_from_embedding(self.act2, self.act1)
		# self.PR = compute_prd_from_embedding(self.act1, self.act2) #Wong
		np.save(self.metricpath+'latest_PR.npy',np.array(self.PR))
		# if self.mode != 'metrics':
		self.PR_vec.append([self.PR,self.total_count.numpy()])

	def print_PR(self):
		path = self.metricpath
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		with PdfPages(path+'PR_plot.pdf') as pdf:
			for PR in self.PR_vec:
				fig1 = plt.figure(figsize=(3.5, 3.5), dpi=400)
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.set_xlim([0, 1])
				ax1.set_ylim([0, 1])
				ax1.get_xaxis().set_visible(True)
				ax1.get_yaxis().set_visible(True)
				precision, recall = PR[0]
				ax1.plot(recall, precision, color = 'g', linestyle = 'solid', alpha=0.5, linewidth=3)
				ax1.set_xlabel('RECALL')
				ax1.set_ylabel('PRECISION')
				title = 'PR at Iteration '+str(PR[1])
				plt.title(title, fontsize=8)
				pdf.savefig(fig1, bbox_inches='tight', dpi=400)
				plt.close(fig1)


	def update_FID(self):
		#FID Funcs vary per dataset. We therefore call the corresponding child func foundin the arch_*.py files
		eval(self.FID_func)

	def eval_FID(self):
		mu1, sigma1 = self.act1.mean(axis=0), cov(self.act1, rowvar=False)
		mu2, sigma2 = self.act2.mean(axis=0), cov(self.act2, rowvar=False)
		# calculate sum squared difference between means
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		covmean = sqrtm(sigma1.dot(sigma2))
		# check and correct imaginary numbers from sqrt
		if iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		self.fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
		self.FID_vec.append([self.fid, self.total_count.numpy()])
		if self.mode == 'metrics':
			print("Final FID score - "+str(self.fid))
			if self.res_flag:
				self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("FID score - "+str(self.fid))

		return


	def print_FID_ACGAN(self):

		np.save(self.metricpath+'FID_even.npy',np.array(self.FID_vec_even))
		np.save(self.metricpath+'FID_odd.npy',np.array(self.FID_vec_odd))
		np.save(self.metricpath+'FID_sharp.npy',np.array(self.FID_vec_sharp))
		np.save(self.metricpath+'FID_single.npy',np.array(self.FID_vec_single))

		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.FID_vec_even)[:,0])
		locs = list(np.array(self.FID_vec_even)[:,1])

		with PdfPages(path+'FID_plot_even.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		vals = list(np.array(self.FID_vec_odd)[:,0])
		locs = list(np.array(self.FID_vec_odd)[:,1])

		with PdfPages(path+'FID_plot_odd.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		vals = list(np.array(self.FID_vec_sharp)[:,0])
		locs = list(np.array(self.FID_vec_sharp)[:,1])

		with PdfPages(path+'FID_plot_sharp.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		vals = list(np.array(self.FID_vec_single)[:,0])
		locs = list(np.array(self.FID_vec_single)[:,1])

		with PdfPages(path+'FID_plot_single.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def print_FID_Rumi(self):

		np.save(self.metricpath+'FID_pos.npy',np.array(self.FID_vec_pos))
		np.save(self.metricpath+'FID_neg.npy',np.array(self.FID_vec_neg))

		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.FID_vec_pos)[:,0])
		locs = list(np.array(self.FID_vec_pos)[:,1])

		with PdfPages(path+'FID_plot_pos.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		vals = list(np.array(self.FID_vec_neg)[:,0])
		locs = list(np.array(self.FID_vec_neg)[:,1])

		with PdfPages(path+'FID_plot_neg.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def print_FID(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.FID_vec)[:,0])
		locs = list(np.array(self.FID_vec)[:,1])

		with PdfPages(path+'FID_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)



	def print_ClassProbs(self):
		path = self.metricpath
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		vals = list(np.array(self.class_prob_vec)[:,0][-1])
		locs = list(np.arange(10))

		with PdfPages(path+'ClassProbs_stem_'+str(self.total_count.numpy())+'.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_ylim([0,0.5])
			ax1.stem(vals,label = 'alpha_p='+str(self.alphap))
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'ClassProbs_plot_'+str(self.total_count.numpy())+'.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'alpha_p='+str(self.alphap))
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)
