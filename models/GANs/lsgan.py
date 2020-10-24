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


###### CLEAN... NEEDS COMMENTS

'''***********************************************************************************
********** LSGAN ELEGANT *************************************************************
***********************************************************************************'''
class LSGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):
		GAN_Base.__init__(self,FLAGS_dict)

	def create_optimizer(self):
		# self.lr_G_scheduled = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=1000, decay_rate=1.0, staircase=True)
		# self.lr_D_scheduled = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_D, decay_steps=500, decay_rate=1.0, staircase=True)
		self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
		self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

		print("Optimizers Successfully made")
		return


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



	def loss_base(self):
		mse = tf.keras.losses.MeanSquaredError()

		D_real_loss = mse(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = mse(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss

		G_fake_loss = mse(tf.ones_like(self.fake_output), self.fake_output)
		self.G_loss = G_fake_loss + D_real_loss 


'''***********************************************************************************
********** SGAN ACGAN ****************************************************************
***********************************************************************************'''
class LSGAN_cGAN(GAN_ACGAN):

	def __init__(self,FLAGS_dict):
		GAN_ACGAN.__init__(self,FLAGS_dict)

	def create_optimizer(self):
		with tf.device(self.device):
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

			print("Optimizers Successfully made")		
		return


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

	def create_optimizer(self):
		with tf.device(self.device):

			self.lr_G_scheduled = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100, decay_rate=0.98, staircase=True)
			self.lr_D_scheduled = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_D, decay_steps=50, decay_rate=0.98, staircase=True)

			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

			print("Optimizers Successfully made")		

		return


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



	def loss_base(self):
		mse = tf.keras.losses.MeanSquaredError()

		D_real_pos_loss = self.alphap * mse(self.label_bp*tf.ones_like(self.real_pos_output), self.real_pos_output)
		D_real_neg_loss = self.alphan * mse(self.label_bn*tf.ones_like(self.real_neg_output), self.real_neg_output)
		D_fake_loss = mse(self.label_a*tf.ones_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_pos_loss + D_real_neg_loss + D_fake_loss 

		G_fake_loss = mse(self.label_c*tf.ones_like(self.fake_output), self.fake_output)
		G_real_neg_loss = mse(self.label_c*tf.ones_like(self.real_neg_output), self.real_neg_output)
		self.G_loss = G_fake_loss + G_real_neg_loss + D_real_pos_loss

