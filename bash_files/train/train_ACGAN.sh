set -e
if [ "${CONDA_DEFAULT_ENV}" != "RumiGAN" ]; then
	echo 'You are not in the <RumiGAN> environment. Attempting to activate the RumiGAN environment. Please run "conda activate RumiGAN" and try again if this fails.'
	condaActivatePath=$(which activate)
	source ${condaActivatePath} RumiGAN
fi


###--- MNIST ---###

python ./gan_main.py  --run_id 'new' --resume 0 --GPU '0' --device '0' --topic 'RumiGAN' --mode 'train' --data 'mnist' --testcase 'few' --number '5' --num_few 200 --gan 'LSGAN' --loss 'base' --saver 1 --num_epochs 100  --res_flag 1 --lr_G 0.00005 --lr_D 0.00005 --paper 1 --batch_size '256' --metrics 'FID,PR' --colab 0 --pbar_flag 1 --latex_plot_flag 0