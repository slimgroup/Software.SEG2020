#!/bin/bash -l

experiment_name=pSGLD_deep_prior
repo_name=Software.SEG2020/siahkoohi2020SEGuqi

path_script=$HOME/$repo_name/src
vel_dir=$HOME/$repo_name/vel_dir

python $path_script/main.py --epoch 10000 --eta 0.01 --lr 0.001 --experiment $experiment_name \
	--weight_decay 200.0 --save_freq 100 --sample_freq 20 \
	--cuda 0 --vel_dir $vel_dir