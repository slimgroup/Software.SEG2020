 #!/bin/bash -l

frequency_old=14.33
frequency_new=14.66
sampling_rate=0.1
scheme=random

experiment_name=ReciprocityGAN_freq${frequency_new}_A_train_${sampling_rate}SamplingRate_${scheme}_evolving_training_set_TransferLearning_from_${frequency_old}
repo_name=Software.SEG2020/zhang2020SEGtli

path_script=$HOME/$repo_name/src/
path_data=$HOME/data
path_model=$HOME/model/$experiment_name
savePath=$path_model/test

python run_plot_data.py --input_data $path_data --save_path $savePath --result_path $path_model/sample --freq $frequency --sampling_scheme $scheme