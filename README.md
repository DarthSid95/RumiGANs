Teaching A GAN What Not to Learn
====================

## Introduction

This is the Code submission accompanying the NeurIPS 2020 paper, titled "[Teaching a GAN What Not to Learn](https://arxiv.org/pdf/2010.15639)."

This code contained the implementations of baseline SGAN, LSGAN, ACGAN, TACGAN, and CGAN-PD employed in the paper to generate the desired results. All code in written in TensorFlow2.x.  Any queries can be directed towards sidartha@iisc.ac.in with [GitHub-RumiGANs] in the subject. In its current state, the code can be used to train new models, while pre-trained models of will be made available shortly.

![RumiGANs](Images/RumiGANs.png?raw=true)

Rumi-GANs split the training data that is input to the GAN discriminator (D) into *positives*, which are target images to model, and *negatives*, which are images from the same dataset, but represent a subset of the distribution to avoid. The generator (G) learns to model only the distribution of the positive samples. Any existing GAN flavor can be reformulated into the *Rumi* framework --- (i) Split the input data into the desirable positive class and the undesirable negative class; (ii) Reformulate the GAN loss to focus on the positive class disribution. The refomulation of all f-GANs and WGAN is presented in the paper.  

The Arxiv version of the paper is available [here](https://arxiv.org/abs/2010.15639), while the NeurIPS 2020 pre-proceedings version of the paper is currently available [here](https://proceedings.neurips.cc/paper/2020/hash/29405e2a4c22866a205f557559c7fa4b-Abstract.html).
## Dependencies and Environment

Dependencies can be installed via anaconda. The ``RumiGAN_GPU.yml`` file list the dependencies to setup the GPU system based environment: 

```
GPU accelerated TensorFlow2.0 Environment:
- pip=20.0.2
- python=3.6.10
- cudatoolkit=10.1.243
- cudnn=7.6.5
- pip:
    - absl-py==0.9.0
    - h5py==2.10.0
    - ipython==7.15.0
    - ipython-genutils==0.2.0
    - matplotlib==3.1.3
    - numpy==1.18.1
    - scikit-learn==0.22.1
    - scipy==1.4.1
    - tensorflow-gpu==2.2.0
    - tensorboard==2.2.0
    - tensorflow-addons
    - tensorflow-estimator==2.2.0
    - tqdm==4.42.1
```
If a GPU is unavailable, the CPU only environment can be built  with ``RumiGAN_CPU.yml``. This setting is meant to run evaluation code. Training is not advisable.
```
CPU based TensorFlow2.0 Environment:
- pip=20.0.2
- python=3.6.10
- pip:
    - absl-py==0.9.0
    - h5py==2.10.0
    - ipython==7.15.0
    - ipython-genutils==0.2.0
    - matplotlib==3.1.3
    - numpy==1.18.1
    - scikit-learn==0.22.1
    - scipy==1.4.1
    - tensorboard==2.0.2
    - tensorflow-addons==0.6.0
    - tensorflow-datasets==3.0.1
    - tensorflow-estimator==2.0.1
    - tensorflow==2.0.0
    - tensorflow-probability==0.8.0
    - tqdm==4.42.1
```

Codes were tested locally on the following system configurations:

```
*SYSTEM 1: Ubuntu 18.04.4LTS
- GPU:			'NVIDIA GeForce GTX 1080'
- RAM:			'32GB'
- CPU:			'Intel Core i7-7820HK @2.9GHz x 8'
- CUDA:			'10.2'
- NVIDIA_drivers:	'440.82' 

*SYSTEM 2: macOS Catalina, Version 10.15.6
- GPU:			-
- RAM:			'16GB'
- CPU:			'8-Core Intel Core i9 @2.3GHz'
- CUDA:			-
- NVIDIA_drivers:	-
```

To create the ``RumiGAN`` environment, run:   
``conda env create -f 'RumiGAN_GPU.yml' `` or ``conda env create -f 'RumiGAN_CPU.yml' ``.

## Training Data

MNIST, Fashion MNIST and CIFAR-10 are loaded from TensorFlow-Datasets. The CelebA dataset (**1.2GB**) can be downloaded by running the following code (requires ``wget`` dependency):

```
python download_celeba.py
```
Alternatively you can manually download the ``img_align_celeba`` folder and the ``list_attr_celeba.csv`` file, and save them at ``RumiGANs/data/CelebA/``.


## Training  

The code provides training procedure for baseline Standard GAN^1, LSGAN^2, WGAN^3, WGAN-GP^4, and each of their corresponding *Rumi* variants. Additionally, ported implementations of auxiliary classifier GAN (ACGAN^5), Twin ACGAN^6 and conditional GAN with projection discriminator (CGAN-PD^7) are included.   


1) The fastest was to train a model is by running the bash files in ``RumiGANs/bash_files/train/``. The train the Model for a given test case: Code to train each ``Figure X Subfig (y)`` is present in these files. Uncomment the desired command to train for the associated testcase. For example, to generate images from Rumi-LSGAN on CelebA with class imbalance, Figure 4(i), uncomment ``Code for Figure 4.i`` in the ``train_RumiGAN.sh`` file in the above folder and run  
```
bash bash_files/train/train_RumiGAN.sh
```
2) Aternatively, you can train any model of your choice by running ``gan_main.py`` with custom flags and modifiers. The list of flags and their defauly values are are defined in  ``gan_main.py``.    

3) **Training on Colab**: This code is capable of training models on Google Colab (although it is *not* optimized for this). For those familiar with the approach, this repository could be cloned to your google drive and steps (1) or (2) could be used for training. CelebA must be downloaded to you current instance on Colab as reading data from GoogleDrive currently causes a Timeout error.  Setting the flags ``--colab_data 1``,  ``--latex_plot_flag 0``, and ``--pbar_flag 0`` is advisable. The ``colab_data`` flag modifies CelebA data-handling code to read data from the local folder, rather than ``RumiGANs/data/CelebA/``.  The ``latex_plot_flag`` flag removes code dependency on latex for plot labels, since the Colab isntance does not native include this. (Alternatively, you could install texlive_full in your colab instance). Lastly, turning off the ``pbar_flag`` was found to prevent the browser from eating too much RAM when training the model. **The .ipynb file for training on Colab will be included shortly**. 

----------------------------------
----------------------------------

### Reference

If you found this code useful, please consider citing our work:

```
@misc{asokan2020teaching,
      title={Teaching a GAN What Not to Learn}, 
      author={Siddarth Asokan and Chandra Sekhar Seelamantula},
      year={2020},
      eprint={2010.15639},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
*The NeurIPS version of the paper is cirrently available in the pre-proceedings*

### License
The license is committed to the repository in the project folder as `LICENSE.txt`.  
Please see the `LICENSE.txt` file for full informations.

----------------------------------

**Siddarth Asokan**  
**Robert Bosch Centre for Cyber Physical Systems **  
**Indian Institute of Science**  
**Bangalore, India **  
**Email:** *siddartha@iisc.ac.in*

----------------------------------
----------------------------------




