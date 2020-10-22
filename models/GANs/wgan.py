from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import tensorflow_probability as tfp
from matplotlib.backends.backend_pgf import PdfPages
tfd = tfp.distributions

import matplotlib.pyplot as plt
import math
import tensorflow as tf
from absl import app
from absl import flags

from gan_topics import *
# tf.keras.backend.set_floatx('float64')


'''***********************************************************************************
********** Baseline WGANs ************************************************************
***********************************************************************************'''
class WGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):

		GAN_Base.__init__(self,FLAGS_dict)

		self.lambda_GP = 0.1 #100 for normal data, 0.1 for synth
		self.lambda_ALP = 10.0 #100 for normal data, 0.1 for synth
		self.lambda_LP = 0.1 #10 for normal? 0.1 for synth
	

	#################################################################
	
	def main_func(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator = eval(self.disc_model)

			print("Model Successfully made")

			print(self.generator.summary())
			print(self.discriminator.summary())

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))


			
			if self.loss == 'GP' :
				self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=200, decay_rate=0.9, staircase=True)
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_schedule, self.beta1, self.beta2)
			elif self.loss == 'ALP' :
				self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100, decay_rate=0.9, staircase=True)
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_schedule, self.beta1, self.beta2)
			else:
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)


			print("Optimizers Successfully made")		

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
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

	#################################################################

	def train(self):    
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1 
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0

			for image_batch in self.train_dataset:
				self.total_count.assign_add(1)
				# batch_count.assign_add(self.Dloop)
				batch_count.assign_add(1)
				start_1 = time.time()
				
				with tf.device(self.device):
					self.train_step(image_batch)
					self.eval_metrics()

				train_time = time.time()-start_1

					
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

			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

		# if self.KLD_flag:
		# 	self.printKLD()

	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		# if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
		# 	self.generate_and_save_batch(epoch)
		# if (self.total_count.numpy() % self.save_step.numpy()) == 0:
		# 	self.generate_and_save_batch(epoch)

	#################################################################

	def test(self):
		self.impath += '_Testing_'
		for img_batch in self.train_dataset:
			self.reals = img_batch
			self.generate_and_save_batch(0)
			return

	#################################################################

	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			with tf.device(self.device):
				noise = tf.random.normal([self.batch_size, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)
			self.reals = reals_all

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				
				self.fakes = self.generator(noise, training=True)

				self.real_output = self.discriminator(self.reals, training = True)
				self.fake_output = self.discriminator(self.fakes, training = True)
				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))

			if self.loss == 'base':
				wt = []
				for w in self.discriminator.get_weights():
					w = tf.clip_by_value(w, -0.1,0.1) #0.01 for [0,1] data, 0.1 for [0,10]
					wt.append(w)
				self.discriminator.set_weights(wt)
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))

	#################################################################

	def loss_base(self):

		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output) 

		self.D_loss = 1 * (-loss_real + loss_fake)

		self.G_loss = 1 * (loss_real - loss_fake)

	#################################################################

	def loss_GP(self):

		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_GP * self.gp 

		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty(self):

		alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		diff = tf.cast(self.fakes,dtype='float32') - tf.cast(self.reals,dtype='float32')
		inter = tf.cast(self.reals,dtype='float32') + (alpha * diff)
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])[0]

		slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
		self.gp = tf.reduce_mean((slopes - 1.)**2)
		return 


	#################################################################

	def loss_LP(self):

		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output)  

		self.lipschitz_penalty()

		self.D_loss = -loss_real + loss_fake + self.lambda_LP * self.lp 

		self.G_loss = loss_real - loss_fake

	def lipschitz_penalty(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py

		self.K = 1
		self.p = 2


		epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1, 1, 1], 0.0, 1.0)
		x_hat = epsilon * self.fakes + (1 - epsilon) * self.reals

		with tf.GradientTape() as t:
			t.watch(x_hat)
			D_vals = self.discriminator(x_hat, training = False)
		grad_vals = t.gradient(D_vals, [x_hat])[0]

		#### args.p taken from github as default p=2
		dual_p = 1 / (1 - 1 / self.p) if self.p != 1 else np.inf

		#gradient_norms = stable_norm(gradients, ord=dual_p)
		grad_norms = tf.norm(grad_vals, ord=dual_p, axis=1, keepdims=True)

		#### Default K = 1
		# lp = tf.maximum(gradient_norms - args.K, 0)
		self.lp = tf.reduce_mean(tf.maximum(grad_norms - self.K, 0)**2)
		# lp_loss = args.lambda_lp * reduce_fn(lp ** 2)

	#################################################################

	def loss_ALP(self):
		
		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output)  

		self.adversarial_lipschitz_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_ALP * self.alp 

		self.G_loss = 1 * (loss_real - loss_fake)


	def adversarial_lipschitz_penalty(self):
		def normalize(x, ord):
			return x / tf.maximum(tf.norm(x, ord=ord, axis=1, keepdims=True), 1e-10)
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		self.eps_min = 0.1
		self.eps_max = 10.0
		self.xi = 10.0
		self.ip = 1
		self.p = 2
		self.K = 5 #was 1. made 5 for G2 compares

		samples = tf.concat([self.reals, self.fakes], axis=0)

		noise = tf.random.uniform([tf.shape(samples)[0], 1, 1, 1], 0, 1)

		eps = self.eps_min + (self.eps_max - self.eps_min) * noise

		with tf.GradientTape(persistent = True) as t:
			t.watch(samples)
			validity = self.discriminator(samples, training = False)

			d = tf.random.uniform(tf.shape(samples), 0, 1) - 0.5
			d = normalize(d, ord=2)
			t.watch(d)
			for _ in range(self.ip):
				samples_hat = tf.clip_by_value(samples + self.xi * d, clip_value_min=-1, clip_value_max=1)
				validity_hat = self.discriminator(samples_hat, training = False)
				dist = tf.reduce_mean(tf.abs(validity - validity_hat))
				grad = t.gradient(dist, [d])[0]
				# print(grad)
				d = normalize(grad, ord=2)
			r_adv = d * eps

		samples_hat = tf.clip_by_value(samples + r_adv, clip_value_min=-1, clip_value_max=1)

		d_lp                   = lambda x, x_hat: tf.norm(x - x_hat, ord=self.p, axis=1, keepdims=True)
		d_x                    = d_lp

		samples_diff = d_x(samples, samples_hat)
		samples_diff = tf.maximum(samples_diff, 1e-10)

		validity      = self.discriminator(samples    , training = False)
		validity_hat  = self.discriminator(samples_hat, training = False)
		validity_diff = tf.abs(validity - validity_hat)

		alp = tf.maximum(validity_diff / samples_diff - self.K, 0)
		# alp = tf.abs(validity_diff / samples_diff - args.K)

		nonzeros = tf.greater(alp, 0)
		count = tf.reduce_sum(tf.cast(nonzeros, tf.float32))

		self.alp = tf.reduce_mean(alp**2)
		# alp_loss = args.lambda_lp * reduce_fn(alp ** 2)

	#####################################################################


'''***********************************************************************************
********** WGAN RumiGAN *************************************************************
***********************************************************************************'''
class WGAN_RumiGAN(GAN_RumiGAN):

	def __init__(self,FLAGS_dict):

		self.lambda_GP = 10.
		
		GAN_RumiGAN.__init__(self,FLAGS_dict)


	def main_func(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator = eval(self.disc_model)

			print("Model Successfully made")

			print(self.generator.summary())
			print(self.discriminator.summary())

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))


			# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100000, decay_rate=0.8, staircase=True)

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
		for epoch in range(start,self.num_epochs):
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

					
				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():6.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:

					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))
					

				if self.total_count.numpy()% self.FID_steps == 0:
					if self.res_flag:
						self.res_file.write("Iteration " + str(self.total_count.numpy()) + '\n')


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
		if self.total_count.numpy() == 1 and 'g' not in self.data:
			predictions = self.reals_pos[0:25]
			path = self.impath + 'pos.png'
			eval(self.show_result_func)
			predictions = self.reals_neg[0:25]
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
			with tf.device(self.device):
				noise = tf.random.normal([self.batch_size, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)
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

			if self.loss == 'base':
				wt = []
				for w in self.discriminator.get_weights():
					w = tf.clip_by_value(w, -0.01,0.01) #0.01 for [0,1] data
					wt.append(w)
				self.discriminator.set_weights(wt)
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))

	def loss_base(self):

		D_fake_loss = tf.reduce_sum(self.fake_output)

		D_real_neg_loss = tf.reduce_sum(self.real_neg_output) 
		D_real_pos_loss = tf.reduce_sum(self.real_pos_output)

		self.D_loss = -1 * (D_real_pos_loss - D_fake_loss) + 0.5 * (D_real_neg_loss - D_fake_loss)

		self.G_loss = - self.D_loss


	def loss_GP(self):

		D_fake_loss = tf.reduce_mean(self.fake_output)

		D_real_neg_loss = tf.reduce_mean(self.real_neg_output) 
		D_real_pos_loss = tf.reduce_mean(self.real_pos_output) 

		self.gradient_penalty()

		self.D_loss =  -1 * self.alphap * (D_real_pos_loss - D_fake_loss) + 1 * self.alphan * (D_real_neg_loss - D_fake_loss) + self.lambda_GP * self.gp_pos + self.lambda_GP * self.gp_neg

		self.G_loss = -self.D_loss


	def gradient_penalty(self):

		alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)

		diff_pos = tf.cast(self.fakes,dtype='float32') - tf.cast(self.reals_pos,dtype='float32')
		inter_pos = tf.cast(self.reals_pos,dtype='float32') + (alpha * diff_pos)
		diff_neg = tf.cast(self.fakes,dtype='float32') - tf.cast(self.reals_neg,dtype='float32')
		inter_neg = tf.cast(self.reals_neg,dtype='float32') + (alpha * diff_neg)

		with tf.GradientTape(persistent = True) as t:
			t.watch(inter_pos)
			pred_pos = self.discriminator(inter_pos, training = True)
			grad_pos = t.gradient(pred_pos, [inter_pos])[0]
			t.reset()
			t.watch(inter_neg)
			pred_neg = self.discriminator(inter_neg, training = True)
			grad_neg = t.gradient(pred_neg, [inter_neg])[0]
		

		slopes_pos = tf.sqrt(tf.reduce_sum(tf.square(grad_pos), axis=[1, 2, 3]))
		self.gp_pos = tf.reduce_mean((slopes_pos - 1.)**2)
		
		slopes_neg = tf.sqrt(tf.reduce_sum(tf.square(grad_neg), axis=[1, 2, 3]))
		self.gp_neg = tf.reduce_mean((slopes_neg - 1.)**2)
		return 

