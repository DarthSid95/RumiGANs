from absl import flags
import os, sys, argparse

FLAGS = flags.FLAGS
FLAGS(sys.argv)

elif FLAGS.topic == 'RumiGAN':
	from .arch_RumiGAN import *
elif FLAGS.topic == 'ACGAN':
	from .arch_ACGAN import *
elif FLAGS.topic == 'cGAN':
	from .arch_cGAN import *
else:
	from .arch_base import *