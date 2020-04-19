#!/bin/bash -l

frequency_old=14.33
frequency_new=14.66
sampling_rate=0.1
scheme=random

experiment_name=ReciprocityGAN_freq${frequency_new}_A_train_${sampling_rate}SamplingRate_${scheme}_evolving_training_set_TransferLearning_from_${frequency_old}
experiment_name_pretraining=ReciprocityGAN_freq${frequency_old}_A_train_${sampling_rate}SamplingRate_${scheme}_evolving_training_set
repo_name=Software.SEG2020/zhang2020SEGtli

path_script=$HOME/$repo_name/paper/src
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

mv $HOME/model/$experiment_name_pretraining $path_model
yes | cp $(dirname "$path_script")/RunTransferLearning.sh $path_model

if [ ! -f $path_data/raw_14.66Hz.hdf5 ]; then
	wget https://www.dropbox.com/s/xp5vv01d2h0fah1/raw_14.66Hz.hdf5 \
		-O $path_data/raw_14.66Hz.hdf5
fi

if [ ! -f $path_data/InterpolatedCoil_freq14.66_A_test_0.1SamplingRate_random.hdf5 ]; then
	wget https://www.dropbox.com/s/tvxdeosfqfp1dbh/InterpolatedCoil_freq14.66_A_test_0.1SamplingRate_random.hdf5 \
		-O $path_data/InterpolatedCoil_freq14.66_A_test_0.1SamplingRate_random.hdf5
fi

if [ ! -f $path_data/InterpolatedCoil_freq14.66_B_test_0.1SamplingRate_random.hdf5 ]; then
	wget https://www.dropbox.com/s/msbdvotiruz2nl0/InterpolatedCoil_freq14.66_B_test_0.1SamplingRate_random.hdf5 \
		-O $path_data/InterpolatedCoil_freq14.66_B_test_0.1SamplingRate_random.hdf5
fi

if [ ! -f $path_data/InterpolatedCoil_freq14.66_Mask_0.1SamplingRate_random.hdf5 ]; then
	wget https://www.dropbox.com/s/dtpezjtq4urf84l/InterpolatedCoil_freq14.66_Mask_0.1SamplingRate_random.hdf5 \
		-O $path_data/InterpolatedCoil_freq14.66_Mask_0.1SamplingRate_random.hdf5
fi

python $path_script/main.py --experiment $experiment_name --phase train --batch_size 1 \
	--freq $frequency_new --data_path $path_data --cuda 1 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample \
	--log_dir $path_model/log --epoch 50 --epoch_step 10 --save_freq 5000  --print_freq 1000 \
