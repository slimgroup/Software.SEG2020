#!/bin/bash -l

frequency=14.33
sampling_rate=0.1
scheme=random

experiment_name=ReciprocityGAN_freq${frequency}_A_train_${sampling_rate}SamplingRate_${scheme}_evolving_training_set
repo_name=Software.SEG2020/zhang2020SEGtli

path_script=$HOME/$repo_name/src/
path_data=$HOME/data
path_model=$HOME/model/$experiment_name

mkdir $HOME/data
mkdir $HOME/model/
mkdir $path_model

if [ ! -f $path_data/raw_14.33Hz.hdf5 ]; then
	wget https://www.dropbox.com/s/66p6dd8g8mdr3i9/raw_14.33Hz.hdf5 \
		-O $path_data/raw_14.33Hz.hdf5
fi

if [ ! -f $path_data/InterpolatedCoil_freq14.33_A_test_0.1SamplingRate_random.hdf5 ]; then
	wget https://www.dropbox.com/s/mnty6achbmk2gzw/InterpolatedCoil_freq14.33_A_test_0.1SamplingRate_random.hdf5 \
		-O $path_data/InterpolatedCoil_freq14.33_A_test_0.1SamplingRate_random.hdf5
fi

if [ ! -f $path_data/InterpolatedCoil_freq14.33_B_test_0.1SamplingRate_random.hdf5 ]; then
	wget https://www.dropbox.com/s/xwf6wen2uifesyh/InterpolatedCoil_freq14.33_B_test_0.1SamplingRate_random.hdf5 \
		-O $path_data/InterpolatedCoil_freq14.33_B_test_0.1SamplingRate_random.hdf5
fi

if [ ! -f $path_data/InterpolatedCoil_freq14.33_Mask_0.1SamplingRate_random.hdf5 ]; then
	wget https://www.dropbox.com/s/tc9bmyqi3r0qeq7/InterpolatedCoil_freq14.33_Mask_0.1SamplingRate_random.hdf5 \
		-O $path_data/InterpolatedCoil_freq14.33_Mask_0.1SamplingRate_random.hdf5
fi

python $path_script/main.py --experiment $experiment_name --phase train --batch_size 1 \
	--freq $frequency --data_path $path_data --cuda 1 \
	--checkpoint_dir $path_model/checkpoint --sample_dir $path_model/sample \
	--log_dir $path_model/log
