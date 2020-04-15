#!/bin/bash -l

experiment_name=deep_prior
repo_name=Software.SEG2020/siahkoohi2020SEGwdp

path_script=$HOME/$repo_name/src
vel_dir=$HOME/$repo_name/vel_dir

python main.py --epoch 3500 --eta 0.01 --lr 0.001 --experiment $experiment_name \
	--weight_decay 2000.0 --save_freq 100 --sample_freq 20 \
	--objective MAP_w --cuda 0 --vel_dir $vel_dir
