from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from absl import app
from absl import flags

from gan_topics import *
			

'''***********************************************************************************
********** SGAN ELeGANt **************************************************************
***********************************************************************************'''
class SGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):
		GAN_Base.__init__(self,FLAGS_dict)

	def main_func(self):

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

			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

			print("Optimizers Successfully made")		


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


	def train(self):    	    
		start = int((self.total_count.numpy() * self.batch_size_big) / (self.train_data.shape[0])) + 1
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
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)
	
	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def test(self):
		self.impath += '_Testing_'
		for img_batch in self.train_dataset:
			self.reals = img_batch
			self.generate_and_save_batch(0)
			return


	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			noise = tf.random.normal([self.batch_size, self.noise_dims], self.noise_mean, self.noise_stddev)
			self.reals = reals_all
			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				self.fakes = self.generator(noise, training=True)

				self.real_output = self.discriminator(self.reals, training=True)
				self.fake_output = self.discriminator(self.fakes, training=True)

				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_base(self):
		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

		D_real_loss = cross_entropy(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = cross_entropy(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss

		G_fake_loss = cross_entropy(tf.ones_like(self.fake_output), self.fake_output)

		self.G_loss = G_fake_loss



'''***********************************************************************************
********** SGAN ACGAN ****************************************************************
***********************************************************************************'''
class SGAN_ACGAN(GAN_ACGAN):

	def __init__(self,FLAGS_dict):
		GAN_ACGAN.__init__(self,FLAGS_dict)

	def main_func(self):
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

			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

			print("Optimizers Successfully made")		

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
				
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 150) == 0:
			self.generate_and_save_batch(epoch)
		# if (self.total_count.numpy() % self.save_step.numpy()) == 0:
		# 	self.generate_and_save_batch(epoch)


	def test(self):
		for i in range(10):
			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			size_figure_grid = self.num_to_print
			test_batch_size = size_figure_grid*size_figure_grid
			noise, noise_labels = self.get_noise('test',test_batch_size)

			if self.label_style == 'base':
				#if base mode, ACGAN generator takes in one_hot labels
				noise_labels = tf.one_hot(np.squeeze(noise_labels), depth = self.num_classes)

			images = self.generator([noise,noise_labels] , training=False)
			if self.data != 'celeba':
				images = (images + 1.0)/2.0
			images_on_grid = self.image_grid(input_tensor = images[0:test_batch_size], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
		# self.impath += '_Testing_'
		# for img_batch in self.train_dataset:
		# 	self.reals = img_batch
		# 	self.generate_and_save_batch(0)
		# 	return


	def train_step(self,reals_all,labels_all):
		for i in tf.range(self.Dloop):
			# noise = tf.random.normal([self.batch_size, self.noise_dims], self.noise_mean, self.noise_stddev)
			self.reals = reals_all
			self.target_labels = labels_all
			self.noise, self.noise_labels = self.get_noise('train', self.batch_size)

			if self.label_style == 'base':
				# if base, Gen takes in one_hot lables "noise_labels_ip", while the loss function uses the regular integer loss "self.noise_labels" and "self.target_labels"
				noise_labels_ip = tf.one_hot(np.squeeze(self.noise_labels), depth = self.num_classes)
			else:
				noise_labels_ip = self.noise_labels

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				self.fakes = self.generator([self.noise,noise_labels_ip] , training=True)

				if self.loss == 'twin':
					self.real_output, self.real_classification, self.real_Cmi = self.discriminator(self.reals, training=True)
					self.fake_output, self.fake_classification, self.fake_Cmi = self.discriminator(self.fakes, training=True)
				else:
					self.real_output, self.real_classification = self.discriminator(self.reals, training=True)
					self.fake_output, self.fake_classification = self.discriminator(self.fakes, training=True)

				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_base(self):
		bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		scce = tf.keras.losses.SparseCategoricalCrossentropy()

		if self.num_classes > 2:
			classifier_loss = scce
		else:
			classifier_loss = bce

		D_real_loss = bce(tf.ones_like(self.real_output), self.real_output)
		D_real_classification = classifier_loss(self.target_labels, self.real_classification)
		D_fake_loss = bce(tf.zeros_like(self.fake_output), self.fake_output)
		D_fake_classification = classifier_loss(self.noise_labels, self.fake_classification)
		self.D_loss = D_real_loss + D_fake_loss + D_real_classification + D_fake_classification

		G_fake_loss = bce(tf.ones_like(self.fake_output), self.fake_output)

		self.G_loss = G_fake_loss + D_fake_classification

	def loss_twin(self):

		bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		# bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits = True)
		scce = tf.keras.losses.SparseCategoricalCrossentropy()
		# ce_logits = tf.nn.softmax_cross_entropy_with_logits()


		if self.num_classes > 2:
			classifier_loss = scce
		else:
			classifier_loss = bce


		D_real_loss = bce(tf.ones_like(self.real_output), self.real_output)
		D_real_classification = classifier_loss(self.target_labels, self.real_classification)

		D_fake_loss = bce(tf.zeros_like(self.fake_output), self.fake_output)
		D_fake_classification = classifier_loss(self.noise_labels, self.fake_classification)
		D_fake_Cmi = classifier_loss(self.noise_labels, self.fake_Cmi)

		self.D_loss = D_real_loss + D_fake_loss + 2*(D_real_classification + D_fake_classification + D_fake_Cmi)

		G_fake_loss = bce(tf.ones_like(self.fake_output), self.fake_output)

		self.G_loss = G_fake_loss + 2*(D_fake_classification - D_fake_Cmi)

'''***********************************************************************************
********** SGAN ACGAN ****************************************************************
***********************************************************************************'''
class SGAN_cGAN(GAN_ACGAN):

	def __init__(self,FLAGS_dict):
		GAN_ACGAN.__init__(self,FLAGS_dict)

	def main_func(self):
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

			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D)

			print("Optimizers Successfully made")		

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


	def train(self):    	    
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)   
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
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
				
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 150) == 0:
			self.generate_and_save_batch(epoch)
		# if (self.total_count.numpy() % self.save_step.numpy()) == 0:
		# 	self.generate_and_save_batch(epoch)


	def test(self):

		for i in range(10):
			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			size_figure_grid = self.num_to_print
			test_batch_size = size_figure_grid*size_figure_grid
			noise, noise_labels = self.get_noise('test',test_batch_size)

			if self.label_style == 'base':
				# If base case, Gen takes in one_hot labels
				noise_labels = tf.one_hot(np.squeeze(noise_labels), depth = self.num_classes)

			images = self.generator([noise,noise_labels] , training=False)
			if self.data != 'celeba':
				images = (images + 1.0)/2.0
			images_on_grid = self.image_grid(input_tensor = images[0:test_batch_size], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()


	def train_step(self,reals_all,labels_all):
		for i in tf.range(self.Dloop):
			# noise = tf.random.normal([self.batch_size, self.noise_dims], self.noise_mean, self.noise_stddev)
			self.reals = reals_all
			self.target_labels = labels_all
			# self.noise_labels  = np.random.randint(0, self.num_classes, self.batch_size)
			# self.noise_labels  = tf.one_hot(np.random.randint(0, self.num_classes, self.batch_size), depth = self.num_classes)
			self.noise, self.noise_labels = self.get_noise('train',self.batch_size)

			if self.label_style == 'base':
				# if base mode, labels needed in one hot for Gen and Disc.
				self.noise_labels = tf.one_hot(np.squeeze(self.noise_labels), depth = self.num_classes)
				self.target_labels =tf.one_hot(np.squeeze(self.target_labels),depth = self.num_classes)

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				self.fakes = self.generator([self.noise,self.noise_labels] , training=True)

				self.real_output = self.discriminator([self.reals,self.target_labels], training=True)
				self.fake_output = self.discriminator([self.fakes,self.noise_labels], training=True)

				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_base(self):
		## NEED FIX
		bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		scce = tf.keras.losses.SparseCategoricalCrossentropy()

		D_real_loss = bce(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = bce(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss 

		G_fake_loss = bce(tf.ones_like(self.fake_output), self.fake_output)

		self.G_loss = G_fake_loss 


	def loss_pd(self):
		## NEED FIX
		bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		scce = tf.keras.losses.SparseCategoricalCrossentropy()

		D_real_loss = bce(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = bce(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss 

		G_fake_loss = bce(tf.ones_like(self.fake_output), self.fake_output)

		self.G_loss = G_fake_loss 


'''***********************************************************************************
********** SGAN RumiGAN **************************************************************
***********************************************************************************'''
class SGAN_RumiGAN(GAN_RumiGAN):

	def __init__(self,FLAGS_dict):
		GAN_RumiGAN.__init__(self,FLAGS_dict)

	def main_func(self):
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

			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

			print("Optimizers Successfully made")		


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
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size_big) / (max(self.train_data_pos.shape[0],self.train_data_neg.shape[0]))) + 1))


	def train(self):    	    
		start = int((self.total_count.numpy() * self.batch_size_big) / (max(self.train_data_pos.shape[0],self.train_data_neg.shape[0]))) + 1
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
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

	def print_batch_outputs(self,epoch):
		if self.total_count.numpy() <= 2 and 'g' not in self.data:
			predictions = self.reals_pos[0:self.num_to_print*self.num_to_print]
			path = self.impath + 'pos.png'
			eval(self.show_result_func)
			predictions = self.reals_neg[0:self.num_to_print*self.num_to_print]
			path = self.impath + 'negs.png'
			eval(self.show_result_func)

		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)


	def test(self):
		self.impath += '_Testing_'
		for img_batch in self.train_dataset:
			self.reals = img_batch
			self.generate_and_save_batch(0)
			return


	def train_step(self, reals_all_pos, reals_all_neg):
		for i in tf.range(self.Dloop):
			noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

			self.reals_pos = reals_all_pos[i*self.batch_size:(i+1)*self.batch_size]
			self.reals_neg = reals_all_neg[i*self.batch_size:(i+1)*self.batch_size]

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				self.fakes = self.generator(noise, training=True)

				self.real_pos_output = self.discriminator(self.reals_pos, training=True)
				self.real_neg_output = self.discriminator(self.reals_neg, training=True)
				self.fake_output = self.discriminator(self.fakes, training=True)

				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	def loss_base(self):
		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

		D_real_pos_loss = tf.constant(self.alphap)*cross_entropy(tf.ones_like(self.real_pos_output), self.real_pos_output)
		#If D_real_neg_loss has ones_like, alphan needs to be pos? - Most likely
		D_real_neg_loss = tf.constant(self.alphan)*cross_entropy(tf.ones_like(self.real_neg_output), self.real_neg_output)
		D_fake_loss = cross_entropy(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_pos_loss + D_real_neg_loss + D_fake_loss 

		self.G_loss = cross_entropy(tf.ones_like(self.fake_output), self.fake_output)



