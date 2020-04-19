#!/bin/bash -l

experiment_name=weak_prior
repo_name=Software.SEG2020/siahkoohi2020SEGwdp

path_script=$HOME/$repo_name/src
vel_dir=$HOME/$repo_name/vel_dir

python $path_script/main.py --epoch 411 --eta 0.01 --lr 0.001 --experiment $experiment_name \
	--Lambda 3000.0 --weight_decay 2000.0 --save_freq 100 --sample_freq 20 \
	--objective weak_MAP_w --cuda 0 --vel_dir $vel_dir
