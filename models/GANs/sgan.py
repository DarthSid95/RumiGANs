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
			
######### CLEAN... NEEDS COMMENTS

'''***********************************************************************************
********** SGAN ELeGANt **************************************************************
***********************************************************************************'''
class SGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):
		GAN_Base.__init__(self,FLAGS_dict)

	def create_optimizer(self):
		with tf.device(self.device):
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)
		print("Optimizers Successfully made")
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
class SGAN_ACGAN(GAN_CondGAN):

	def __init__(self,FLAGS_dict):
		GAN_CondGAN.__init__(self,FLAGS_dict)

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
class SGAN_cGAN(GAN_CondGAN):

	def __init__(self,FLAGS_dict):
		GAN_CondGAN.__init__(self,FLAGS_dict)

	def create_optimizer(self):
		with tf.device(self.device):
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

			print("Optimizers Successfully made")		
		return


	def train_step(self,reals_all,labels_all):
		for i in tf.range(self.Dloop):
			self.reals = reals_all
			self.target_labels = labels_all
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
		bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		scce = tf.keras.losses.SparseCategoricalCrossentropy()

		D_real_loss = bce(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = bce(tf.zeros_like(self.fake_output), self.fake_output)
		self.D_loss = D_real_loss + D_fake_loss 

		G_fake_loss = bce(tf.ones_like(self.fake_output), self.fake_output)

		self.G_loss = G_fake_loss 


	def loss_pd(self):
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

	def create_optimizer(self):
		with tf.device(self.device):
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)

		print("Optimizers Successfully made")
		return



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



