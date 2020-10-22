# from __future__ import print_function
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


# tf.keras.backend.set_floatx('float64')

'''***********************************************************************************
********** LSGAN ELEGANT *************************************************************
***********************************************************************************'''
class LSGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):
		GAN_Base.__init__(self,FLAGS_dict)

	
	def main_func(self):
		with tf.device(self.device):
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

			# self.lr_schedule_G = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=1000, decay_rate=1.0, staircase=True)
			# self.lr_schedule_D = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_D, decay_steps=500, decay_rate=1.0, staircase=True)

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




				
						
			if self.pbar_flag:
				bar.close()
				del bar
			# tf.print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)



	def print_batch_outputs(self,epoch):

		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)

		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)



	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)
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



	def test(self):
		for i in range(self.num_test_images):
			path = self.impath+'_Testing2_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			size_figure_grid = self.num_to_print
			test_batch_size = size_figure_grid*size_figure_grid
			noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

			images = self.generator(noise, training=False)
			if self.data != 'celeba':
				images = (images + 1.0)/2.0
			images_on_grid = self.image_grid(input_tensor = images[0:test_batch_size], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(14,14))
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


	def loss_base(self):
		mse = tf.keras.losses.MeanSquaredError()

		D_real_loss = mse(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss

		G_fake_loss = mse(tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 


	def loss_logdiff(self):
		mse = tf.keras.losses.MeanSquaredError()

		D_real_loss = mse(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss

		G_fake_loss = mse(tf.ones_like(self.fake_output), self.fake_output)
		G_sim_loss = -tf.reduce_mean(tf.math.log(tf.reduce_sum(tf.reduce_sum(tf.math.abs(tf.subtract(self.fakes[0:-1],self.fakes[1:])), axis = 1), axis = 1)))
		# print(G_sim_loss)
		self.G_loss = G_fake_loss + D_real_loss + G_sim_loss


	def loss_intD(self):
		mse = tf.keras.losses.MeanSquaredError()

		D_real_loss = mse(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(tf.zeros_like(self.fake_output), self.fake_output)

		self.D_loss = D_real_loss + D_fake_loss
		D_loss_integral_real = tf.reduce_sum(self.real_output, axis = 0)
		D_loss_integral_fake = tf.reduce_sum(self.fake_output, axis = 0)

		G_fake_loss = mse(tf.ones_like(self.fake_output), self.fake_output)

		self.G_loss = G_fake_loss + D_real_loss - (D_loss_integral_fake[0])# + D_loss_integral_real[0])


'''***********************************************************************************
********** SGAN ACGAN ****************************************************************
***********************************************************************************'''
class LSGAN_cGAN(GAN_ACGAN):

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
			self.noise, self.noise_labels = self.get_noise('train',self.batch_size)

			if self.label_style == 'base':
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
		mse = tf.keras.losses.MeanSquaredError()

		D_real_loss = mse(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss

		G_fake_loss = mse(tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 


	def loss_pd(self):
		mse = tf.keras.losses.MeanSquaredError()

		D_real_loss = mse(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss

		G_fake_loss = mse(tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 

'''***********************************************************************************
********** LSGAN RumiGAN *************************************************************
***********************************************************************************'''
class LSGAN_RumiGAN(GAN_RumiGAN):

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

			self.lr_schedule_G = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100, decay_rate=0.98, staircase=True)
			self.lr_schedule_D = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_D, decay_steps=50, decay_rate=0.98, staircase=True)

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
		start = int((self.total_count.numpy() * self.batch_size_big) / (self.max_data_size)) + 1 
		for epoch in range(start,self.num_epochs+1):
			if self.pbar_flag:
				bar = self.pbar(epoch) 
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0
			for image_batch_pos,image_batch_neg in zip(self.train_dataset_pos,self.train_dataset_neg):

				self.total_count.assign_add(1)
				batch_count.assign_add(self.Dloop)
				start_1 = time.time()
				with tf.device(self.device):
					self.train_step(image_batch_pos,image_batch_neg)
					self.eval_metrics()
				train_time = time.time()-start_1

				# if self.res_flag:
				# 	self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():3.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.3e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.3e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					# tf.print ('Epoch {:>3d} batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f}'.format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))

				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()
					# tf.print("Model and Images saved")

			if self.pbar_flag:
				bar.close()
				del bar
			tf.print('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)


			# writer.flush

	def print_batch_outputs(self,epoch):
		if self.total_count.numpy() <= 2 and 'g' not in self.data:
			predictions = self.reals_pos[0:self.num_to_print*self.num_to_print]
			if self.data!='celeba':
				predictions = (predictions + 1.0)/(2.0)
			path = self.impath + 'pos.png'
			eval(self.show_result_func)
			predictions = self.reals_neg[0:self.num_to_print*self.num_to_print]
			if self.data!='celeba':
				predictions = (predictions + 1.0)/(2.0)
			path = self.impath + 'negs.png'
			eval(self.show_result_func)
		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	def train_step(self, reals_all_pos, reals_all_neg):
		for i in tf.range(self.Dloop):
			noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

			self.reals_pos = reals_all_pos
			self.reals_neg = reals_all_neg

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

	def test(self):
		for i in range(self.num_test_images):
			path = self.impath+'_Testing4_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			size_figure_grid = self.num_to_print
			test_batch_size = size_figure_grid*size_figure_grid
			noise = tf.random.normal([self.batch_size, self.noise_dims],self.noise_mean, self.noise_stddev)

			images = self.generator(noise, training=False)
			if self.data != 'celeba':
				images = (images + 1.0)/2.0
			images_on_grid = self.image_grid(input_tensor = images[0:test_batch_size], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(25,25))
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

	def loss_base(self):
		#Set1
		# self.alpha_pos = tf.constant(1.5)
		# self.alpha_neg = tf.constant(1.)

		#Set2
		# self.alpha_pos = tf.constant(0.5)
		# self.alpha_neg = tf.constant(1.)

		mse = tf.keras.losses.MeanSquaredError()

		D_real_pos_loss = self.alphap * mse(self.label_bp*tf.ones_like(self.real_pos_output), self.real_pos_output)
		D_real_neg_loss = self.alphan * mse(self.label_bn*tf.ones_like(self.real_neg_output), self.real_neg_output)
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_pos_loss + D_real_neg_loss + D_fake_loss 

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output)
		G_real_neg_loss = mse(self.label_c*tf.ones_like(self.real_neg_output), self.real_neg_output)
		self.G_loss = G_fake_loss + G_real_neg_loss + D_real_pos_loss

		##Colab 0 and 1 use c = +2 and -2 resp, on set1. local 0 uses c = a = -1, and sed2


	def loss_intD(self):
		mse = tf.keras.losses.MeanSquaredError()

		D_real_loss = mse(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(tf.zeros_like(self.fake_output), self.fake_output)

		self.D_loss = D_real_loss + D_fake_loss
		D_loss_integral_real = tf.reduce_sum(self.real_output, axis = 0)
		D_loss_integral_fake = tf.reduce_sum(self.fake_output, axis = 0)

		G_fake_loss = mse(tf.ones_like(self.fake_output), self.fake_output)

		self.G_loss = G_fake_loss + D_real_loss - (D_loss_integral_real[0] + D_loss_integral_fake[0])


