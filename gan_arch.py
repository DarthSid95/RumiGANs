from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json
import glob
from sklearn.manifold import TSNE
from tqdm.autonotebook import tqdm
import shutil

import tensorflow_probability as tfp
tfd = tfp.distributions
# import tensorflow_gan as tfgan
# import prd_score as prd

##FOR FID
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
import scipy as sp
from numpy import iscomplexobj
# from gan_data import *
# from arch_mnist import *
# from arch_u1 import *


####NOTE : 15thJune2020 - Cleaned up KLD calculator. sorted folder creation ops here. Need to remove them elsewheres. Need to clean print KLD,FID function.

# FLAGS = flags.FLAGS
# FLAGS(sys.argv)
# # tf.keras.backend.set_floatx('float64')

# if FLAGS.loss == 'deq':
# 	from arch/arch_deq import *
# elif FLAGS.topic == 'CycleGAN':
# 	from arch_CycleGAN import *
# elif FLAGS.topic == 'RumiGAN':
# 	from arch_RumiGAN import *
# else:
from arch import *
from ops import *

'''
GAN_ARCH Consists of the common parts of GAN architectures, speficially, the calls to the sub architecture classes from the respective files, and the calls for FID evaluations. Each ARCH_data_* class has archtectures corresponding to that dataset learning and for the loss case ( Autoencoder structures for DEQ case, etc.)
'''
'''***********************************************************************************
********** GAN Arch ******************************************************************
***********************************************************************************'''
class GAN_ARCH(eval('ARCH_'+FLAGS.data)): #mnist, ARCH_celeba, ARCG_g1, ARCH_g2, ARCH_gmm8, ARCH_comma):  eval('ARCH_'+FLAGS.data),

	def __init__(self,FLAGS_dict):
		''' Defines anything common to te diofferent GAN approaches. Architectures of Gen and Disc, all flags,'''
		for name,val in FLAGS_dict.items():
			exec('self.'+name+' = val')
		# self.topic = topic
		# self.data = data
		# self.loss = loss
		# self.gan_name = gan
		# self.mode = mode
		# self.colab = colab

		if self.colab and (self.data in ['mnist', 'celeba', 'cifar10']):
			self.bar_flag = 0
		else:
			self.bar_flag = 1

		# self.metrics = metrics
	
		# self.num_parallel_calls = num_parallel_calls
		# self.saver = saver
		# self.resume = resume
		# self.res_flag = res_file
		# self.save_all = save_all_models
		# self.paper = paper

		if self.device == '-1':
			self.device = '/CPU'
		elif self.device == '-2':
			self.device = '/TPU'
		elif self.device == '':
			self.device = '/GPU'
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

		''' Sort out local special conditions. Higher learning rates for 1DG and DEQ. If printing for paper PDF, set up saving KLD for gaussians. '''
		if self.data in ['g1', 'g2', 'gmm2']:
			self.batch_size = tf.constant(500, dtype='int64')
		if self.data in ['gmm8' ]:
			self.batch_size = tf.constant(500, dtype='int64')
		

		eval('ARCH_'+FLAGS.data+'.__init__(self)')

		if self.data in ['g1', 'g2', 'gmm8', 'gmm2']:
			self.num_to_print = 1000
		else:
			self.num_to_print = 10

		if self.mode in ['test','metrics']:
			self.num_test_images = 20
		else:
			self.num_test_images = 10
		# if self.topic in['ACGAN', 'cGAN']:
		# 	### Need to FIX
		# 	self.num_to_print = 10
			# if self.data != 'celeba':
			# 	self.num_to_print = self.num_classes**2
			# else:
			# 	self.num_to_print = self.num_classes*50



		if self.topic == 'AAE':	
			self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.3e}', 2: f'{0:2.3e}', 3: f'{0:2.3e}'}
			self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining} Time: {elapsed}  D_Loss: {postfix[1]} G_Loss: {postfix[2]} AE_Loss: {postfix[3]}'
		else:
			self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.3e}', 2: f'{0:2.3e}'}
			self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining} Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'

		# self.log_dir = 'logs/NIPS_May2020/RumiGAN_Cifar10_Compare/'
		# self.log_dir = 'logs/NIPS_May2020/RumiGAN_Fashion_Compare/RandomClasses/NewJUne/'#'logs/NIPS_May2020/RumiGAN_MNIST_Compares/Singles5MAny/' #'logs/NIPS_May2020/RumiGAN_CelebA_Compare/Males/'#'logs/NIPS_Mar2020/RumiGAN_Compares/Compare_Sharps/'

		### NEEDS FIXING...NEEDS A FLAG
		# log_dir = 'logs/Log_Folder_03072020/'
		# log_dir = None
		if self.log_folder == 'default':
			today = date.today()
			self.log_dir = 'logs/Log_Folder_'+today.strftime("%d%m%Y")+'/'
		else:
			self.log_dir = self.log_folder#'logs/NIPS_June2020/'
		
		if self.log_dir[-1] != '/':
			self.log_dir += '/'	

		self.run_id_flag = self.run_id
		
		self.create_run_location()
		self.setup_metrics()


		self.timestr = time.strftime("%Y%m%d-%H%M%S")
		if self.res_flag == 1:
			self.res_file = open(self.run_loc+'/'+self.run_id+'_Results.txt','a')
			FLAGS.append_flags_into_file(self.run_loc+'/'+self.run_id+'_Flags.txt')
			# js = json.dumps(flags.FLAGS.flag_values_dict())
			# f.write(js)
			# f.close()


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
		#### AAE are Autoencoders, not generative models.
		if self.topic == 'AAE':
			predictions = self.Decoder(self.Encoder(self.reals[0:self.num_to_print*self.num_to_print], training = False), training = False)
		elif self.topic in ['cGAN', 'ACGAN']:
			class_vec = []
			for i in range(self.num_classes):
				class_vec.append(i*np.ones(int((self.num_to_print**2)/self.num_classes)))
			class_final = np.expand_dims(np.concatenate(class_vec,axis = 0),axis = 1)
			if self.label_style == 'base':
				class_final = tf.one_hot(np.squeeze(class_final), depth = self.num_classes)
			predictions = self.generator([noise,class_final], training=False)
		else:
			predictions = self.generator(noise, training=False)
			if self.loss == 'FS':
				if self.latent_kind in ['AE','AAE']:
					predictions = self.Decoder(predictions, training= False)
		# if self.mode == 'test':
		# 	self.fakes = predictions
		# if not self.paper:
		if self.topic == 'AAE' or self.data != 'celeba':
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
		self.KLD_flag = 0
		self.FID_flag = 0
		self.PR_flag = 0
		self.lambda_flag = 0
		self.recon_flag = 0
		self.GradGrid_flag = 0
		self.class_prob_flag = 0
		self.metric_counter_vec = []


		if self.loss == 'FS' and self.mode != 'metrics':
			self.lambda_flag = 1
			self.lambda_vec = []

		if 'KLD' in self.metrics:				
			self.KLD_flag = 1
			self.KLD_vec = []

			if self.data in ['g1', 'g2', 'gmm2', 'gmm8', 'gN', 'u1']:
				self.KLD_steps = 10
				if self.data == 'gN':
					self.KLD_steps = 50
				if self.data in ['gmm2', 'gmm8', 'u1']:#, 'gN']:
					self.KLD_func = self.KLD_sample_estimate
				else:
					self.KLD_func = self.KLD_Gaussian
			else:
				self.KLD_flag = 1
				self.KLD_steps = 100
				if self.loss == 'FS' and self.topic != 'AAE':
					if self.distribution == 'gaussian' or self.data in ['g1','g2']:
						self.KLD_func = self.KLD_Gaussian
					else:
						self.KLD_func = self.KLD_sample_estimate
				if self.topic == 'AAE':
					if 'gaussian' in self.noise_kind:
						self.KLD_func = self.KLD_Gaussian
					else:
						self.KLD_func = self.KLD_sample_estimate
				print('KLD is not an accurate metric on this datatype')
				

		if 'FID' in self.metrics:
			self.FID_flag = 1
			self.FID_load_flag = 0
			self.FID_vec = []
			self.FID_vec_new = []

			if self.data in ['mnist']:
				self.FID_steps = 500
				if self.mode == 'metrics':
					self.FID_num_samples = 50000
				else:
					self.FID_num_samples = 5000#15000
			elif self.data in ['cifar10']:
				self.FID_steps = 500
				if self.mode == 'metrics':
					self.FID_num_samples = 50000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['celeba']:
				self.FID_steps = 2500
				if self.mode == 'metrics':
					self.FID_num_samples = 50000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['gN']:
				self.FID_steps = 100
			else:
				self.FID_flag = 0
				print('FID cannot be evaluated on this dataset')

		if 'recon' in self.metrics:
			self.recon_flag = 1
			self.recon_vec = []
			self.FID_vec_new = []

			if self.data in ['mnist']:
				self.recon_steps = 500
			elif self.data in ['cifar10']:
				self.recon_steps = 1500
			elif self.data in ['celeba']:
				self.recon_steps = 1500
			elif self.data in ['gN']:
				self.recon_steps = 100
			else:
				self.recon_flag = 0
				print('Reconstruction cannot be evaluated on this dataset')


		if 'PR' in self.metrics:
			### NEed to DeisGN
			self.PR_flag = 1
			self.PR_vec = []
			self.PR_steps = self.FID_steps

		if 'GradGrid' in self.metrics:
			if self.data in ['g2', 'gmm8']:
				self.GradGrid_flag = 1
				self.GradGrid_steps = 100
			else:
				print("Cannot plot Gradient grid. Not a 2D dataset")

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



		if self.KLD_flag and ((self.total_count.numpy()%self.KLD_steps == 0 or self.total_count.numpy() == 1)  or self.mode == 'metrics'):
			update_flag = 1
			self.update_KLD()
			if self.mode != 'metrics':
				np.save(self.metricpath+'KLD.npy',np.array(self.KLD_vec))
				self.print_KLD()

		if self.recon_flag and ((self.total_count.numpy()%self.recon_steps == 0 or self.total_count.numpy() == 1)  or self.mode == 'metrics'):
			update_flag = 1
			self.eval_recon()
			if self.mode != 'metrics':
				np.save(self.metricpath+'recon.npy',np.array(self.recon_vec))
				self.print_recon()


		if self.lambda_flag and (self.loss == 'FS' or self.mode == 'metrics'):
			update_flag = 1
			self.update_Lambda()
			if self.mode != 'metrics':
				np.save(self.metricpath+'Lambda.npy',np.array(self.lambda_vec))
				self.print_Lambda()

		if self.GradGrid_flag and ((self.total_count.numpy()%self.GradGrid_steps == 0 or self.total_count.numpy() == 1) or self.mode == 'metrics'):
			update_flag = 1
			self.print_GradGrid()

		if self.class_prob_flag and (self.total_count.numpy()%self.class_prob_steps == 0 or self.mode == 'metrics'):
			update_flag = 1
			self.class_prob_metric()
			if self.mode != 'metrics':
				np.save(self.metricpath+'ClassProbs.npy',np.array(self.class_prob_vec))
				self.print_ClassProbs()

		if self.res_flag and update_flag:
			self.res_file.write("Metrics avaluated at Iteration " + str(self.total_count.numpy()) + '\n')


	# def update_FLOPS(self):

	# 	def get_flops(model_h5_path):
	# 		session = tf.compat.v1.Session()
	# 		graph = tf.compat.v1.get_default_graph()
					

	# 		with graph.as_default():
	# 			with session.as_default():
	# 				model = tf.keras.models.load_model(model_h5_path)

	# 				run_meta = tf.compat.v1.RunMetadata()
	# 				opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
				
	# 				# We use the Keras session graph in the call to the profiler.
	# 				flops = tf.compat.v1.profiler.profile(graph=graph,
	# 													  run_meta=run_meta, cmd='op', options=opts)
				
	# 				return flops.total_float_ops

	# 	return 

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


	def eval_FID_new(self):

		def symmetric_matrix_square_root(mat, eps=1e-10):
			"""Compute square root of a symmetric matrix.
			Note that this is different from an elementwise square root. We want to
			compute M' where M' = sqrt(mat) such that M' * M' = mat.
			Also note that this method **only** works for symmetric matrices.
			Args:
			mat: Matrix to take the square root of.
			eps: Small epsilon such that any element less than eps will not be square
			rooted to guard against numerical instability.
			Returns:
			Matrix square root of mat.
			"""
			# Unlike numpy, tensorflow's return order is (s, u, v)

			s, u, v = tf.linalg.svd(mat)
			# sqrt is unstable around 0, just use 0 in such case
			si = tf.compat.v1.where(tf.less(s, eps), s, tf.sqrt(s))
			# Note that the v returned by Tensorflow is v = V
			# (when referencing the equation A = U S V^T)
			# This is unlike Numpy which returns v = V^T
			# print(u.shape,si.shape,v.shape)
			return tf.matmul(tf.matmul(u, tf.linalg.tensor_diag(si)), v, transpose_b=True)


		def trace_sqrt_product(sigma, sigma_v):
			"""Find the trace of the positive sqrt of product of covariance matrices.
			'_symmetric_matrix_square_root' only works for symmetric matrices, so we
			cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
			('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).
			Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
			We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
			Note the following properties:
			(i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
			 => eigenvalues(A A B B) = eigenvalues (A B B A)
			(ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
			 => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
			(iii) forall M: trace(M) = sum(eigenvalues(M))
			 => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
										   = sum(sqrt(eigenvalues(A B B A)))
										   = sum(eigenvalues(sqrt(A B B A)))
										   = trace(sqrt(A B B A))
										   = trace(sqrt(A sigma_v A))
			A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
			use the _symmetric_matrix_square_root function to find the roots of these
			matrices.
			Args:
			sigma: a square, symmetric, real, positive semi-definite covariance matrix
			sigma_v: same as sigma
			Returns:
			The trace of the positive square root of sigma*sigma_v
			"""

			# Note sqrt_sigma is called "A" in the proof above
			sqrt_sigma = symmetric_matrix_square_root(sigma)

			# This is sqrt(A sigma_v A) above
			sqrt_a_sigmav_a = tf.matmul(sqrt_sigma, tf.matmul(sigma_v, sqrt_sigma))

			if iscomplexobj(sqrt_a_sigmav_a):
				sqrt_a_sigmav_a = sqrt_a_sigmav_a.real

			return tf.linalg.trace(symmetric_matrix_square_root(sqrt_a_sigmav_a))


		m = (tf.reduce_mean(input_tensor=self.act1, axis=0),)
		m_w = (tf.reduce_mean(input_tensor=self.act2, axis=0),)
		# Calculate the unbiased covariance matrix of first activations.
		num_examples_real = tf.cast(tf.shape(input=self.act1)[0], tf.float64)
		sigma = tf.cast(num_examples_real / (num_examples_real - 1),dtype = tf.float64) * tf.cast(tfp.stats.covariance(self.act1),dtype = tf.float64)
		# Calculate the unbiased covariance matrix of second activations.
		num_examples_generated = tf.cast(tf.shape(input=self.act2)[0], tf.float64)
		sigma_w = tf.cast(num_examples_generated / (num_examples_generated - 1),dtype = tf.float64) * tf.cast(tfp.stats.covariance(self.act2),dtype = tf.float64)

		# print(sigma.shape, sigma_w.shape)
		sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

		# Compute the two components of FID.

		# First the covariance component.
		# Here, note that trace(A + B) = trace(A) + trace(B)
		trace = tf.linalg.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

		# Next the distance between means.
		mean = tf.reduce_sum(input_tensor=tf.math.squared_difference(m, m_w))  # Equivalent to L2 but more stable.
		fid = tf.cast(trace, tf.float64) + tf.cast(mean, tf.float64)
		self.fid = tf.cast(fid, tf.float64)
		# print(fid)
		self.FID_vec.append([fid.numpy(), self.total_count.numpy()])
		if self.mode == 'metrics':
			print("Final FID score - "+str(self.fid))
			if self.res_flag:
				self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("FID score - "+str(self.fid))


		return fid

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
		# self.eval_FID_new()
		# np.save(self.impath+'_FID.npy',np.array(self.FID_vec))
		return


	def print_FID_ACGAN(self):

		# np.save(self.metricpath+'FID_even.npy',np.array(self.FID_vec_even))
		# np.save(self.metricpath+'FID_odd.npy',np.array(self.FID_vec_odd))
		# np.save(self.metricpath+'FID_sharp.npy',np.array(self.FID_vec_sharp))
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

		# vals = list(np.array(self.FID_vec_even)[:,0])
		# locs = list(np.array(self.FID_vec_even)[:,1])

		# with PdfPages(path+'FID_plot_even.pdf') as pdf:

		# 	fig1 = plt.figure(figsize=(3.5, 3.5))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.get_xaxis().set_visible(True)
		# 	ax1.get_yaxis().set_visible(True)
		# 	ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
		# 	ax1.legend(loc = 'upper right')
		# 	pdf.savefig(fig1)
		# 	plt.close(fig1)

		# vals = list(np.array(self.FID_vec_odd)[:,0])
		# locs = list(np.array(self.FID_vec_odd)[:,1])

		# with PdfPages(path+'FID_plot_odd.pdf') as pdf:

		# 	fig1 = plt.figure(figsize=(3.5, 3.5))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.get_xaxis().set_visible(True)
		# 	ax1.get_yaxis().set_visible(True)
		# 	ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
		# 	ax1.legend(loc = 'upper right')
		# 	pdf.savefig(fig1)
		# 	plt.close(fig1)

		# vals = list(np.array(self.FID_vec_sharp)[:,0])
		# locs = list(np.array(self.FID_vec_sharp)[:,1])

		# with PdfPages(path+'FID_plot_sharp.pdf') as pdf:

		# 	fig1 = plt.figure(figsize=(3.5, 3.5))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.get_xaxis().set_visible(True)
		# 	ax1.get_yaxis().set_visible(True)
		# 	ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
		# 	ax1.legend(loc = 'upper right')
		# 	pdf.savefig(fig1)
		# 	plt.close(fig1)

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
		# if self.topic == 'ELeGANt':
		# 	if self.loss == 'FS' and self.latent_kind == 'AE':
		# 		basis = np.expand_dims(np.array(np.arange(0,self.total_count.numpy() - self.AE_steps,self.FID_steps)),axis=1)
		# 	else:
		# 		basis = np.expand_dims(np.array(np.arange(0,self.total_count.numpy(),self.FID_steps)),axis=1)
		# else:
		# 	basis  = np.expand_dims(np.array(np.arange(0,len(self.FID_vec),self.FID_steps)),axis=1)

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

		# vals = list(np.array(self.FID_vec_new)[:,0])
		# locs = list(np.array(self.FID_vec_new)[:,1])

		# with PdfPages(path+'FID_plot_new.pdf') as pdf:

		# 	fig1 = plt.figure(figsize=(3.5, 3.5))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.get_xaxis().set_visible(True)
		# 	ax1.get_yaxis().set_visible(True)
		# 	ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
		# 	ax1.legend(loc = 'upper right')
		# 	pdf.savefig(fig1)
		# 	plt.close(fig1)

	def eval_recon(self):
		# print('Evaluating Recon Loss\n')
		mse = tf.keras.losses.MeanSquaredError()
		for image_batch in self.recon_dataset:
			# print("batch 1\n")
			recon_images = self.Decoder(self.Encoder(image_batch, training= False) , training = False)
			try:
				recon_loss = 0.5*(recon_loss) + 0.25*tf.reduce_mean(tf.abs(image_batch - recon_images)) + 0.75*(mse(image_batch,recon_images))
			except:
				recon_loss = 0.5*tf.reduce_mean(tf.abs(image_batch - recon_images)) + 1.5*(mse(image_batch,recon_images))
		self.recon_vec.append([recon_loss, self.total_count.numpy()])
		if self.mode == 'metrics':
			print("Final Reconstruction error - "+str(recon_loss))
			if self.res_flag:
				self.res_file.write("Final Reconstruction error - "+str(recon_loss))

		if self.res_flag:
			self.res_file.write("Reconstruction error - "+str(recon_loss))


	def print_recon(self):
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

		vals = list(np.array(self.recon_vec)[:,0])
		locs = list(np.array(self.recon_vec)[:,1])

		with PdfPages(path+'recon_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'Reconstruction Error')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


	def KLD_sample_estimate(self,P,Q):
		def skl_estimator(s1, s2, k=1):
			from sklearn.neighbors import NearestNeighbors
			### Code Courtesy nheartland 
			### URL : https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py
			""" KL-Divergence estimator using scikit-learn's NearestNeighbours
			s1: (N_1,D) Sample drawn from distribution P
			s2: (N_2,D) Sample drawn from distribution Q
			k: Number of neighbours considered (default 1)
			return: estimated D(P|Q)
			"""
			n, m = len(s1), len(s2)
			d = float(s1.shape[1])
			D = np.log(m / (n - 1))

			s1_neighbourhood = NearestNeighbors(k+1, 10).fit(s1)
			s2_neighbourhood = NearestNeighbors(k, 10).fit(s2)

			for p1 in s1:
				s1_distances, indices = s1_neighbourhood.kneighbors([p1], k+1)
				s2_distances, indices = s2_neighbourhood.kneighbors([p1], k)
				rho = s1_distances[0][-1]
				nu = s2_distances[0][-1]
				D += (d/n)*np.log(nu/rho)
			return D
		KLD = skl_estimator(P,Q)
		self.KLD_vec.append([KLD, self.total_count.numpy()])
		return

	def KLD_Gaussian(self,P,Q):

		def get_mean(f):
			return np.mean(f,axis = 0).astype(np.float64)
		def get_cov(f):
			return np.cov(f,rowvar = False).astype(np.float64)
		def get_std(f):
			return np.std(f).astype(np.float64)
		try:
			if self.data == 'g1':
				Distribution = tfd.Normal
				P_dist = Distribution(loc=get_mean(P), scale=get_std(P))
				Q_dist = Distribution(loc=get_mean(Q), scale=get_std(Q))
			else:
				Distribution = tfd.MultivariateNormalFullCovariance
				P_dist = Distribution(loc=get_mean(P), covariance_matrix=get_cov(P))
				Q_dist = Distribution(loc=get_mean(Q), covariance_matrix=get_cov(Q))
		
			self.KLD_vec.append([P_dist.kl_divergence(Q_dist).numpy(), self.total_count.numpy()])
		except:
			print("KLD error - Falling back to prev value")
			try:
				self.KLD_vec.append([self.KLD_vec[-1]*0.9, self.total_count.numpy()])
			except:
				self.KLD_vec.append([0, self.total_count.numpy()])
		# print('KLD: ',self.KLD_vec[-1])
		return


	def update_KLD(self):
		
		if self.topic == 'ELeGANt':
			if self.loss == 'FS' and (self.latent_kind == 'AE' or self.latent_kind == 'AAE'):
				self.KLD_func(self.reals_enc,self.fakes_enc)
			else:
				self.KLD_func(self.reals,self.fakes)
		elif self.topic == 'AAE':
			self.KLD_func(self.fakes_enc,self.reals_enc)
		else:
			self.KLD_func(self.reals,self.fakes)


		# if self.topic == 'ELeGANt':
		# 	if self.loss == 'deq' and (self.latent_kind == 'AE' or self.latent_kind == 'AAE'):
		# 		pd_dist = MultiVarNormal(loc=get_mean(self.reals_enc), covariance_matrix=get_cov(self.reals_enc))
		# 		pg_dist = MultiVarNormal(loc=get_mean(self.fakes_enc), covariance_matrix=get_cov(self.fakes_enc))
		# 	else:
		# 		pd_dist = MultiVarNormal(loc=get_mean(self.reals), covariance_matrix=get_cov(self.reals))
		# 		pg_dist = MultiVarNormal(loc=get_mean(self.fakes), covariance_matrix=get_cov(self.fakes))
		# elif self.topic == 'AAE':
		# 	pg_dist = MultiVarNormal(loc=get_mean(self.reals_enc), covariance_matrix=get_cov(self.reals_enc))
		# 	pd_dist = MultiVarNormal(loc=get_mean(self.fakes_enc), covariance_matrix=get_cov(self.fakes_enc))
		# else:
		# 	pd_dist = MultiVarNormal(loc=get_mean(self.reals), covariance_matrix=get_cov(self.reals))
		# 	pg_dist = MultiVarNormal(loc=get_mean(self.fakes), covariance_matrix=get_cov(self.fakes))


		# print(pd_dist.kl_divergence(pg_dist))
		# print(X)

			

	def print_KLD(self):
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
		# if self.topic == 'ELeGANt':
		# 	if self.loss == 'deq' and self.latent_kind == 'AE':
		# 		basis = np.expand_dims(np.array(np.arange(0,self.total_count.numpy() - self.AE_steps)),axis=1)
		# else:
		# 	basis = np.expand_dims(np.array(np.arange(0,self.total_count.numpy(),self.KLD_steps)),axis=1)

		# basis  = np.expand_dims(np.array(np.arange(0,len(self.KLD))),axis=1)*self.KLD_steps
		vals = list(np.array(self.KLD_vec)[:,0])
		locs = list(np.array(self.KLD_vec)[:,1])
		if self.topic == 'ELeGANt':
			if self.loss == 'FS' and self.latent_kind == 'AE':
				locs = list(np.array(self.KLD_vec)[:,1] - self.AE_steps)
		

		with PdfPages(path+'KLD_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'KL Divergence Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def update_Lambda(self):
		self.lambda_vec.append([self.lamb.numpy(),self.total_count.numpy()])

	def print_Lambda(self):
		path = self.metricpath
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size": 12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		# lbasis  = np.expand_dims(np.array(np.arange(0,len(self.lambda_vec))),axis=1)
		vals = list(np.array(self.lambda_vec)[:,0])
		locs = list(np.array(self.lambda_vec)[:,1])

		with PdfPages(path+'Lambda_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'Lambda Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


	# def update_GradGrid(self):



	def print_GradGrid(self):

		path = self.metricpath + str(self.total_count.numpy()) + '_'

		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size": 12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		
		from itertools import product as cart_prod

		x = np.arange(self.MIN,self.MAX+0.1,0.1)
		y = np.arange(self.MIN,self.MAX+0.1,0.1)

		# X, Y = np.meshgrid(x, y)
		prod = np.array([p for p in cart_prod(x,repeat = 2)])
		# print(x,prod)

		X = prod[:,0]
		Y = prod[:,1]

		# print(prod,X,Y)
		# print(XXX)

		with tf.GradientTape() as disc_tape:
			prod = tf.cast(prod, dtype = 'float32')
			disc_tape.watch(prod)
			d_vals = self.discriminator(prod,training = False)
		grad_vals = disc_tape.gradient(d_vals, [prod])[0]

		# print(d_vals, prod)
		### Theres a swap. hence the traspose and the 1,0 assignment of dx and dy
		try:
			# print(d_vals[0])
			
			if False and ((min(d_vals[0]) <= -2) or (max(d_vals[0]) >= 2)):
				### IF NORMALIZATION IS NEEDED
				d_vals_sub = d_vals[0] - min(d_vals[0])
				d_vals_norm = d_vals_sub/max(d_vals_sub)
				d_vals_norm -= 0.5
				d_vals_norm *= 3
				# d_vals_new = np.expand_dims(np.array(d_vals_norm),axis = 1)
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()
				# d_vals_norm = np.expand_dims(np.array(d_vals_sub/max(d_vals_sub)),axis = 1)
				# d_vals_new = np.subtract(d_vals_norm,0.5)
				# d_vals_new = np.multiply(d_vals_new,3.)
				# print(d_vals_new)
			else:
				### IF NORMALIZATION IS NOT NEEDED
				d_vals_norm = d_vals[0]
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()
		except:
			d_vals_new = np.reshape(d_vals,(x.shape[0],y.shape[0])).transpose()
		# print(d_vals_new)
		dx = grad_vals[:,1]
		dy = grad_vals[:,0]
		# print(XXX)
		n = -1
		color_array = np.sqrt(((dx-n)/2)**2 + ((dy-n)/2)**2)

		with PdfPages(path+'GradGrid_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=self.MIN,top=self.MAX)
			ax1.quiver(X,Y,dx,dy,color_array)
			ax1.scatter(self.reals[:1000,0], self.reals[:1000,1], c='r', linewidth = 1, label='Real Data', marker = '.', alpha = 0.1)
			ax1.scatter(self.fakes[:1000,0], self.fakes[:1000,1], c='g', linewidth = 1, label='Fake Data', marker = '.', alpha = 0.1)
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'Contourf_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			cs = ax1.contourf(x,y,d_vals_new,alpha = 0.5, levels = list(np.arange(-1.5,1.5,0.1)), extend = 'both' )
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			# Can be used with figure size (2,10) to generate a colorbar with diff colors as plotted
			# Good for a table wit \multicol{5}
			# cbar = fig1.colorbar(cs, aspect = 40, shrink=1., ticks = [0, 1.0], orientation = 'horizontal')
			# cbar.ax.set_xticklabels(['Min', 'Max'])
			# # cbar.set_ticks_position(['bottom', 'top'])
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'Contourf_plot_cBAr.pdf') as pdf:

			fig1 = plt.figure(figsize=(8, 8))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			cs = ax1.contourf(x,y,d_vals_new,alpha = 0.5, levels = list(np.arange(-1.5,1.6,0.1)), extend = 'both' )
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			# Can be used with figure size (10,2) to generate a colorbar with diff colors as plotted
			# Good for a table wit \multicol{5}
			cbar = fig1.colorbar(cs, aspect = 40, shrink=1., ticks = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5], orientation = 'horizontal')
			cbar.ax.set_xticklabels(['$-1.5$', '$-1$', '$-0.5$', '$0$', '$0.5$', '$1$', '$1.5$'])
			# # cbar.set_ticks_position(['bottom', 'top'])
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'Contour_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			ax1.contour(x,y,d_vals_new,10,linewidths = 0.5,alpha = 0.4 )
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
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

	# def updateKLD(self):
	# 	if self.topic == 'ELeGANt':
	# 		if self.loss == 'deq' and self.latent_kind == 'AE' or self.latent_kind == 'AAE':
	# 			pd_dist = tfd.MultivariateNormalFullCovariance(loc=np.mean(self.reals_enc, axis = 0), covariance_matrix=np.cov(self.reals_enc, rowvar = False).astype(np.float32))
	# 			pg_dist = tfd.MultivariateNormalFullCovariance(loc=np.mean(self.fakes_enc, axis = 0), covariance_matrix=np.cov(self.fakes_enc,rowvar = False).astype(np.float32))
	# 			cov = (0.25*(np.cov(self.reals_enc, rowvar = False) + np.cov(self.fakes_enc, rowvar = False))).astype(np.float32)
	# 			lc = (0.5*(np.mean(self.fakes_enc, axis = 0) +  np.mean(self.reals_enc, axis = 0))).astype(np.float32) 
	# 			pm_dist = tfd.MultivariateNormalFullCovariance(loc=lc, covariance_matrix = cov)
	# 	elif self.topic == 'AAE':
	# 		pg_dist = tfd.MultivariateNormalFullCovariance(loc=np.mean(self.reals_enc, axis = 0), covariance_matrix=np.cov(self.reals_enc,rowvar = False).astype(np.float32))
	# 		pd_dist = tfd.MultivariateNormalFullCovariance(loc=np.mean(self.fakes_enc, axis = 0), covariance_matrix=np.cov(self.fakes_enc,rowvar = False).astype(np.float32))
	# 		cov = (0.25*(np.cov(self.reals_enc, rowvar = False) + np.cov(self.fakes_enc, rowvar = False))).astype(np.float32)
	# 		lc = (0.5*(np.mean(self.fakes_enc, axis = 0) +  np.mean(self.reals_enc, axis = 0))).astype(np.float32) 
	# 		pm_dist = tfd.MultivariateNormalFullCovariance(loc=lc, covariance_matrix = cov)
	# 	else:
	# 		pd_dist = tfd.MultivariateNormalFullCovariance(loc=np.mean(self.reals, axis = 0), covariance_matrix=np.cov(self.reals,rowvar = False).astype(np.float32))
	# 		pg_dist = tfd.MultivariateNormalFullCovariance(loc=np.mean(self.fakes, axis = 0), covariance_matrix=np.cov(self.fakes,rowvar = False).astype(np.float32))
	# 		cov = (0.25*(np.cov(self.reals, rowvar = False) + np.cov(self.fakes, rowvar = False))).astype(np.float32)
	# 		lc = (0.5*(np.mean(self.fakes, axis = 0) +  np.mean(self.reals, axis = 0))).astype(np.float32) 
	# 		pm_dist = tfd.MultivariateNormalFullCovariance(loc=lc, covariance_matrix = cov)
	# 	self.KLD.append(0.5*pd_dist.kl_divergence(pm_dist) + 0.5*pg_dist.kl_divergence(pm_dist))	

	# def updateKLD(self):
	# 	if self.topic == 'ELeGANt':
	# 		if self.loss == 'deq' and self.latent_kind == 'AE':
	# 			pd_dist = tfd.MultivariateNormalDiag(loc=np.mean(self.reals_enc).astype(np.float32), scale_diag=np.diagonal(np.cov(self.reals_enc)).astype(np.float32))
	# 			pg_dist = tfd.MultivariateNormalDiag(loc=np.mean(self.fakes_enc).astype(np.float32), scale_diag=np.diagonal(np.cov(self.fakes_enc)).astype(np.float32))
	# 			# sc = np.diagonal(0.5*(np.cov(self.reals_enc) + np.cov(self.fakes_enc))).astype(np.float32)
	# 			# lc = (0.5*(np.mean(self.fakes_enc) +  np.mean(self.reals_enc))).astype(np.float32) 
	# 			# pm_dist = tfd.MultivariateNormalDiag(loc=lc, scale_diag = sc)
	# 	elif self.topic == 'AAE':
	# 		pg_dist = tfd.MultivariateNormalDiag(loc=np.mean(self.reals_enc).astype(np.float32), scale_diag=np.diagonal(np.cov(self.reals_enc)).astype(np.float32))
	# 		pd_dist = tfd.MultivariateNormalDiag(loc=np.mean(self.fakes_enc).astype(np.float32), scale_diag=np.diagonal(np.cov(self.fakes_enc)).astype(np.float32))
	# 		# sc = np.diagonal(0.5*(np.cov(self.reals_enc) + np.cov(self.fakes_enc))).astype(np.float32)
	# 		# lc = (0.5*(np.mean(self.fakes_enc) +  np.mean(self.reals_enc))).astype(np.float32)
	# 		# pm_dist = tfd.MultivariateNormalDiag(loc=lc, scale_diag = sc)
	# 	else:
	# 		pd_dist = tfd.MultivariateNormalDiag(loc=np.mean(self.reals), scale_diag=np.diagonal(np.cov(self.reals)))
	# 		pg_dist = tfd.MultivariateNormalDiag(loc=np.mean(self.fakes), scale_diag=np.diagonal(np.cov(self.fakes)))
	# 		# sc = np.diagonal(0.5*(np.cov(self.reals_enc) + np.cov(self.fakes_enc))).astype(np.float32)
	# 		# lc = (0.5*(np.mean(self.fakes_enc) +  np.mean(self.reals_enc))).astype(np.float32) 
	# 		# pm_dist = tfd.MultivariateNormalDiag(loc=lc, scale_diag = sc)


	# 	self.KLD.append(pd_dist.kl_divergence(pg_dist))	

