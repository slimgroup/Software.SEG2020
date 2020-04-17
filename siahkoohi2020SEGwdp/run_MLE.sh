#!/bin/bash -l

experiment_name=MLE
repo_name=Software.SEG2020/siahkoohi2020SEGwdp

path_script=$HOME/$repo_name/src
vel_dir=$HOME/$repo_name/vel-model

python $path_script/main.py --epoch 411 --eta 0.01 --lr 0.001 --experiment $experiment_name \
	--save_freq 100 --sample_freq 20 --objective MLE_x --cuda 0 --vel_dir $vel_dir
