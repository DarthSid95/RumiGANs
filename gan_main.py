from __future__ import print_function
import os, sys, time, argparse, signal, json, struct
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.python import debug as tf_debug
import traceback

print(tf.__version__)
from absl import app
from absl import flags



# from mnist_cnn_icp_eval import *
# tf.keras.backend.set_floatx('float64')

def signal_handler(sig, frame):
	print('\n\n\nYou pressed Ctrl+C! \n\n\n')
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

'''Generic set of FLAGS. learning_rate and batch_size are redefined in GAN_ARCH if g1/g2'''
FLAGS = flags.FLAGS
flags.DEFINE_float('lr_G', 0.0001, """learning rate for generator""")
flags.DEFINE_float('lr_D', 0.0001, """learning rate for discriminator""")
flags.DEFINE_float('beta1', 0.5, """beta1 for Adam""")
flags.DEFINE_float('beta2', 0.9, """beta2 for Adam""")
flags.DEFINE_list('metrics', '', 'CSV for the metrics to evaluate. KLD, FID, PR')
flags.DEFINE_integer('batch_size', 100, """Batch size.""")
flags.DEFINE_integer('seed', 42, """Initialize the random seed of the run (for reproducibility).""")
flags.DEFINE_integer('num_epochs', 200, """Number of epochs to train for.""")
flags.DEFINE_integer('Dloop', 1, """Number of loops to run for D.""")
flags.DEFINE_integer('Gloop', 1, """Number of loops to run for G.""")
flags.DEFINE_integer('num_parallel_calls', 5, """Number of parallel calls for dataset map function""")

flags.DEFINE_integer('colab', 0, """ set 1 to run code in a colab friendy way """)
flags.DEFINE_integer('latex_plot_flag', 0, """set 1 for plots comptible with latex syntax in text fields""")
flags.DEFINE_integer('pbar_flag', 1, """1-Display Progress Bar, 0 O.W.""")

flags.DEFINE_integer('paper', 1, """1 for saving images for a paper""")
flags.DEFINE_integer('resume', 1, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_integer('saver', 1, """1-Save events for Tensorboard. 0 O.W.""")
flags.DEFINE_integer('res_flag', 1, """1-Write results to a file. 0 O.W.""")
flags.DEFINE_integer('out_size', 32, """CelebA output reshape size""")
flags.DEFINE_integer('save_all', 0, """1-Save all the models. 0 for latest 10""") #currently functions as save_all internally

flags.DEFINE_string('run_id', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('log_folder', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('mode', 'train', """Operation mode: train, test, fid """)

flags.DEFINE_string('topic', 'RumiGAN', """GAN methodology to run: Base, ACGAN, cGAN, or RumiGAN""")
flags.DEFINE_string('data', 'mnist', """Type of data to run on: mnist, fmnist, celeba, cifar10 """)
flags.DEFINE_string('gan', 'sgan', """GAN variant to use: SGAN, LSGAN, WGAN """)
flags.DEFINE_string('loss', 'base',"""Type of loss function to use: base, twin(ACGAN), pd(cGAN), GP(WGAN)""")
flags.DEFINE_string('GPU', '0,1', """GPU's made visible '0', '1', or '0,1' """)
flags.DEFINE_string('device', '0', """Input to tf.device(): Which device to run on: 0,1 or -1(CPU)""")


'''Flags just for RumiGAN Paper analysis'''
flags.DEFINE_integer('number', 3, """ Class selector in Multi-class data on mnist/fmnist/cifar10""")
flags.DEFINE_integer('num_few', 200, """Num of images for minority 200((F)MNIST), 1k(Cifar10), 5k(CelebA)""")
flags.DEFINE_string('label_style', 'base', """Label input style to cGAN/ACGANs :base/embed/multiply""")

flags.DEFINE_float('label_a', -0.5, """Class label - a """)
flags.DEFINE_float('label_bp', 2.0, """Class label - bp for +ve data """)
flags.DEFINE_float('label_bn', -2.0, """Class label - bn for -ve data """)
flags.DEFINE_float('label_c', 2.0, """Class label - c for generator """)

flags.DEFINE_float('alphap', 2.5, """alpha_plus/beta_plus weight for +ve class loss term """)
flags.DEFINE_float('alphan', 0.5, """alpha_minus/beta_minus weight for -ve class loss term""")

flags.DEFINE_string('testcase', 'female', """Test cases for RumiGAN""")
'''
Defined Testcases:
MNIST/FMNIST:
1. even - even numbers as positive class
2. odd - odd numbers as positive class
3. overlap - "Not true random - determinitic to the set selected in the paper" 
4. rand - 6 random classes as positive, 6 as negative
5. single - learn a single digit in MNIST - uses "number" flag to deice which number
6. few - learn a single digit (as minority positive) in MNIST - uses "number" flag to deice which number, "num_few" to decide how many samples to pick for minority class 
CelebA:
1. male - learn males in CelebA as positive
2. female - learn females in CelebA as positive
3. fewmale - learn males as minority positive class in CelebA - "num_few" used as in MNIST.6
4. fewfemale - learn females as minority positive class in CelebA - "num_few" used as in MNIST.6
5. hat - learn hat in CelebA as positive
6. bald - learn bald in CelebA as positive
CIFAR-10:
1. single - as in MNIST
2. few - as in MNIST
3. animals - learn animals as positive class, vehicles as negative
'''


FLAGS(sys.argv)
from models import *


if __name__ == '__main__':
	'''Enable Flags and various tf declarables on GPU processing '''
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	print('Visible Physical Devices: ',physical_devices)
	for gpu in physical_devices:
		print(gpu)
		tf.config.experimental.set_memory_growth(gpu, True)
	tf.config.threading.set_inter_op_parallelism_threads(6)
	tf.config.threading.set_intra_op_parallelism_threads(6)

	
	# Level | Level for Humans | Level Description                  
	# ------|------------------|------------------------------------ 
	# 0     | DEBUG            | [Default] Print all messages       
	# 1     | INFO             | Filter out INFO messages           
	# 2     | WARNING          | Filter out INFO & WARNING messages 
	# 3     | ERROR            | Filter out all messages
	tf.get_logger().setLevel('ERROR')
	# tf.debugging.set_log_device_placement(True)
	# if FLAGS.colab and FLAGS.data == 'celeba':
	# os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "500G"

	# if FLAGS.colab:
	# 	import warnings
	# 	warnings.filterwarnings("ignore")

	''' Set random seed '''
	np.random.seed(FLAGS.seed)
	tf.random.set_seed(FLAGS.seed)

	FLAGS_dict = FLAGS.flag_values_dict()

	###	EXISTING Variants:
	##
	##	(1) SGAN - 
	##		(A) Base-- loss variants: base
	##		(B) RumiGAN -- loss variants: base
	##		(C) ACGAN -- loss variants: base, twin
	##		(D) cGAN -- loss variants: pd
	##
	##	(2) LSGAN - 
	##		(A) Base -- loss variants: base
	##		(B) RumiGAN -- loss variants: base
	##		(C) cGAN -- loss variants: pd
	##
	##	(3) WGAN - 
	##		(A) Base -- loss variants: base, GP
	##		(C) Rumi -- loss variants: base, GP
	##
	##
	### -----------------

	gan_call = FLAGS.gan + '_' + FLAGS.topic + '(FLAGS_dict)'

	print('GAN Training will commence')
	gan = eval(gan_call)
	gan.initial_setup()
	gan.main_func()
	print('Worked')

	if gan.mode == 'train':
		print(gan.mode)
		gan.train()
		gan.test()

	if gan.mode == 'h5_from_checkpoint':
		gan.h5_from_checkpoint()

	if gan.mode == 'test':
		gan.test()

	if gan.mode == 'metrics':
		gan.eval_metrics()


###############################################################################  
	
	
	print('Completed.')
