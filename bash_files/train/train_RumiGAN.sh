set -e
if [ "${CONDA_DEFAULT_ENV}" != "RumiGAN" ]; then
	echo 'You are not in the <RumiGAN> environment. Attempting to activate the RumiGAN environment. Please run "conda activate RumiGAN" and try again if this fails.'
	condaActivatePath=$(which activate)
	source ${condaActivatePath} RumiGAN
fi

###--- MNIST ---###

# python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'mnist' --testcase 'few' --number '5' --num_few 75 --saver 1  --res_flag 1 --num_epochs 100  --lr_G 0.00005 --lr_D 0.00005 --paper 1 --batch_size '256' --metrics 'FID,PR' --colab 0 --pbar_flag 1 --latex_plot_flag 0

# python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'mnist' --testcase 'even'  --saver 1 --res_flag 1 --num_epochs 75  --lr_G 0.00005 --lr_D 0.00005 --paper 1 --batch_size '256' --metrics 'FID,PR' --colab 0 --pbar_flag 1 --latex_plot_flag 0

# python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'mnist' --testcase 'overlap' --saver 1 --res_flag 1 --num_epochs 75  --lr_G 0.00005 --lr_D 0.00005 --paper 1 --batch_size '256' --metrics 'FID,PR' --colab 0 --pbar_flag 1 --latex_plot_flag 0

###--- Fashion MNIST ---###

python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'mnist' --mnist_variant 'fashion' --testcase 'overlap' --saver 1 --res_flag 1 --num_epochs 75  --lr_G 0.0002 --lr_D 0.0002 --paper 1 --batch_size '256' --metrics 'FID,PR' --colab 0 --pbar_flag 1 --latex_plot_flag 0

###--- CelebA ---###

# python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'celeba' --testcase 'hat' --out_size 128  --saver 1 --num_epochs 200  --res_flag 1 --lr_G 0.0002 --lr_D 0.0002 --paper 1 --batch_size '100' --metrics 'FID,PR'  --colab 0 --pbar_flag 1 --latex_plot_flag 0

# python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'celeba' --testcase 'bald' --out_size 128 --saver 1 --num_epochs 200  --res_flag 1 --lr_G 0.0002 --lr_D 0.0002 --paper 1 --batch_size '100' --metrics 'FID,PR'  --colab 0 --pbar_flag 1 --latex_plot_flag 0

# python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'celeba' --testcase 'fewmale' --num_few 10000 --out_size 128 --num_few 10000  --saver 1 --num_epochs 200  --res_flag 1 --lr_G 0.0002 --lr_D 0.0002 --paper 1 --batch_size '100' --metrics 'FID,PR'  --colab 0 --pbar_flag 1 --latex_plot_flag 0

# python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'celeba' --testcase 'fewfemale' --num_few 10000 --out_size 128 --num_few 10000  --saver 1 --num_epochs 200  --res_flag 1 --lr_G 0.0002 --lr_D 0.0002 --paper 1 --batch_size '100' --metrics 'FID,PR'  --colab 0 --pbar_flag 1 --latex_plot_flag 0

###--- CIFAR-10 ---###

# python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'cifar10' --testcase 'few' --number '7' --num_few 250 --saver 1  --res_flag 1 --num_epochs 100  --lr_G 0.00005 --lr_D 0.00005 --paper 1 --batch_size '256' --metrics 'FID,PR' --colab 0 --pbar_flag 1 --latex_plot_flag 0

# python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --gan 'LSGAN' --loss 'base' --mode 'train' --data 'cifar10' --testcase 'animals'  --saver 1 --res_flag 1 --num_epochs 250  --lr_G 0.00005 --lr_D 0.00005 --paper 1 --batch_size '256' --metrics 'FID,PR' --colab 0 --pbar_flag 1 --latex_plot_flag 0

