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

# import tensorflow_probability as tfp
# tfd = tfp.distributions

##FOR FID
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
import scipy as sp
from numpy import iscomplexobj

from ext_resources import *

class GAN_Metrics():

	def __init__(self):

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
		
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages


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
			
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

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
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

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
		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages

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

		if self.latex_plot_flag:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})
		else:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex = False)

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
