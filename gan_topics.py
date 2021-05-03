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

from gan_data import *
from gan_src import *

# import tensorflow_probability as tfp
# tfd = tfp.distributions
from matplotlib.backends.backend_pgf import PdfPages



'''
GAN_topic is the Overarching class file, where corresponding parents are instantialized, along with setting up the calling functions for these and files and folders for resutls, etc. data reading is also done from here. Sometimes display functions, architectures, etc may be modified here if needed (overloading parent classes)
'''

'''***********************************************************************************
********** GAN Baseline setup ********************************************************
***********************************************************************************'''
class GAN_Base(GAN_SRC, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all fundamental ops and metric functions'''
		GAN_SRC.__init__(self,FLAGS_dict)
		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'


		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()

		# self.create_models()

		# self.create_optimizer()

		# self.create_load_checkpoint()

	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data = eval(self.gen_func)

		self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		self.train_dataset_size = self.train_data.shape[0]

		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size,  self.num_batches,self.print_step, self.save_step))

	def create_models(self):

		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator = eval(self.disc_model)

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

			print("Model Successfully made")

			print(self.generator.summary())
			print(self.discriminator.summary())
		return		

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 D_optimizer = self.D_optimizer,
								 generator = self.generator,
								 discriminator = self.discriminator,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size_big) / (self.train_data.shape[0])) + 1))
		return

	def train(self):
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch in self.train_dataset:
				# print(image_batch.shape)
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.time()
				with tf.device(self.device):
					self.train_step(image_batch)
					self.eval_metrics()
				train_time = time.time()-start_time

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():6.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))

				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
	
	def print_batch_outputs(self,epoch):

		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def test(self):
		for i in range(self.num_test_images):

			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())

			size_figure_grid = self.num_to_print
			test_batch_size = size_figure_grid*size_figure_grid
			noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

			images = self.generator(noise, training=False)
			if self.data != 'celeba':
				images = (images + 1.0)/2.0

			self.save_image_batch(images = images,label = label, path = path)

		# self.impath += '_Testing_'
		# for img_batch in self.train_dataset:
		# 	self.reals = img_batch
		# 	self.generate_and_save_batch(0)
		# 	return

'''***********************************************************************************
********** Conditional GAN (cGAN-PD, ACGAN, TACGAN) setup ****************************
***********************************************************************************'''
class GAN_CondGAN(GAN_SRC, GAN_DATA_CondGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''

		GAN_SRC.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_CondGAN.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.train_labels, self.batch_size)'
		# self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		if self.loss == 'FS':
			self.gen_model = 'self.generator_model_'+self.data+'_'+self.latent_kind+'()'
			self.disc_model = 'self.discriminator_model_'+self.data+'_'+self.latent_kind+'()' 
			self.EncDec_func = 'self.encoder_model_'+self.data+'_'+self.latent_kind+'()'
			self.DEQ_func = 'self.discriminator_ODE()'

		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()

		# self.create_models()

		# self.create_optimizer()

		# self.create_load_checkpoint()

	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data, self.train_labels = eval(self.gen_func)

		self.num_batches = int(np.floor((self.train_data.shape[0])/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		print("Dataset created - this is it")
		print(self.train_dataset)

		self.train_dataset_size = self.train_data.shape[0]

		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, 
		 self.num_batches,self.print_step, self.save_step))

	def get_noise(self,noise_case,batch_size):
		noise = tf.random.normal([batch_size, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)
		if noise_case == 'test':
			if self.data in ['mnist', 'cifar10']:
				if self.testcase in ['single', 'few']:
					noise_labels = self.number*np.ones((batch_size,1)).astype('int32')
				elif self.testcase in ['sharp']:
					noise_labels = np.expand_dims(np.random.choice([1,2,4,5,7,9], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['even']:
					noise_labels = np.expand_dims(np.random.choice([0,2,4,6,8], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['odd']:
					noise_labels = np.expand_dims(np.random.choice([1,3,5,7,9], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['animals']:
					noise_labels = np.expand_dims(np.random.choice([2,3,4,5,6,7], batch_size), axis = 1).astype('int32')
			elif self.data in ['celeba']:
				if self.testcase in ['male', 'fewmale', 'bald', 'hat']:
					noise_labels = np.ones((batch_size,1)).astype('int32')
				elif self.testcase in ['female', 'fewfemale']:
					noise_labels = np.zeros((batch_size,1)).astype('int32')
		if noise_case == 'train':
			noise_labels = np.random.randint(0, self.num_classes, batch_size)
			if self.data == 'celeba':
				noise_labels = np.expand_dims(noise_labels, axis = 1)

		return noise, noise_labels

	def create_models(self):

		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator = eval(self.disc_model)

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

			print("Model Successfully made")

			print(self.generator.summary())
			print(self.discriminator.summary())
		return		

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 D_optimizer = self.D_optimizer,
								 generator = self.generator,
								 discriminator = self.discriminator,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size_big) / (self.train_data.shape[0])) + 1))
		return

	def train(self):
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.time()
			batch_count = tf.Variable(0, dtype='int64')
			start_time = 0

			for image_batch,labels_batch in self.train_dataset:
				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_time = time.time()
				with tf.device(self.device):
					self.train_step(image_batch,labels_batch)
					self.eval_metrics()
				train_time = time.time()-start_time

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():6.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))
				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()

				if (self.total_count.numpy() % 1000) == 0:
					self.test()


			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

	def print_batch_outputs(self,epoch):
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def test(self):
		for i in range(10):

			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())

			size_figure_grid = self.num_to_print
			test_batch_size = size_figure_grid*size_figure_grid
			noise, noise_labels = self.get_noise('test',test_batch_size)

			if self.label_style == 'base':
				#if base mode, ACGAN generator takes in one_hot labels
				noise_labels = tf.one_hot(np.squeeze(noise_labels), depth = self.num_classes)

			images = self.generator([noise,noise_labels] , training=False)
			if self.data != 'celeba':
				images = (images + 1.0)/2.0
			
			self.save_image_batch(images = images,label = label, path = path)

'''***********************************************************************************
********** GAN RumiGAN setup *********************************************************
***********************************************************************************'''
class GAN_RumiGAN(GAN_SRC, GAN_DATA_RumiGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''
		GAN_SRC.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_RumiGAN.__init__(self)


	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data_pos, self.train_data_neg, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		''' Define dataset and tf.data function. batch sizing done'''
		# self.get_data()

		# self.create_models()

		# self.create_optimizer()

		# self.create_load_checkpoint()

	def get_data(self):
		
		with tf.device('/CPU'):
			self.train_data_pos, self.train_data_neg = eval(self.gen_func)
			self.max_data_size = max(self.train_data_pos.shape[0],self.train_data_neg.shape[0])

			self.num_batches = int(np.floor(self.max_data_size/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset_pos, self.train_dataset_neg = eval(self.dataset_func)

			self.train_dataset_size = self.max_data_size
		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, 
		 self.num_batches,self.print_step, self.save_step))

	def create_models(self):

		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator = eval(self.disc_model)

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

			print("Model Successfully made")

			print(self.generator.summary())
			print(self.discriminator.summary())
		return		

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 D_optimizer = self.D_optimizer,
								 generator = self.generator,
								 discriminator = self.discriminator,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size_big) / (self.train_data.shape[0])) + 1))
		return

	def train(self):    	    
		start = int((self.total_count.numpy() * self.batch_size) / (max(self.train_data_pos.shape[0],self.train_data_neg.shape[0]))) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
			start_time =0

			for image_batch_pos,image_batch_neg in zip(self.train_dataset_pos,self.train_dataset_neg):

				self.total_count.assign_add(1)
				batch_count.assign_add(self.Dloop)
				start_time = time.time()
				with tf.device(self.device):
					self.train_step(image_batch_pos,image_batch_neg)
					self.eval_metrics()
				train_time = time.time()-start_time

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():6.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))

				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

	def print_batch_outputs(self,epoch):
		if self.total_count.numpy() <= 2 and 'g' not in self.data:
			predictions = self.reals_pos[0:self.num_to_print*self.num_to_print]
			if self.data!='celeba':
				predictions = (predictions + 1.0)/(2.0)
			path = self.impath + 'pos.png'
			label = 'POSITIVE CLASS SAMPLES'
			self.save_image_batch(images = predictions,label = label, path = path)
			# eval(self.show_result_func)
			predictions = self.reals_neg[0:self.num_to_print*self.num_to_print]
			if self.data!='celeba':
				predictions = (predictions + 1.0)/(2.0)
			path = self.impath + 'negs.png'
			label = "NEGATIVE CLASS SAMPLES"
			self.save_image_batch(images = predictions,label = label, path = path)
			# eval(self.show_result_func)
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def test(self):
		for i in range(self.num_test_images):

			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())

			size_figure_grid = self.num_to_print
			test_batch_size = size_figure_grid*size_figure_grid
			noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

			images = self.generator(noise, training=False)
			if self.data != 'celeba':
				images = (images + 1.0)/2.0
			self.save_image_batch(images = images,label = label, path = path)