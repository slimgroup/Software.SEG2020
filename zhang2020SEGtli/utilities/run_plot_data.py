import numpy as np
import argparse
from plot_data import plotting_class
np.random.seed(seed=19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--freq', dest='frequency', type=float, default=14.66, 
	help='frequency of data')
parser.add_argument('--sampling_rate', dest='sampling_rate', type=float, default=0.1, 
	help='sampling rate')
parser.add_argument('--input_data', dest='input_data', type=str, default='/home/ec2-user/data', 
	help='path to input data')
parser.add_argument('--save_path', dest='save_path', type=str, default='/home/ec2-user/model', 
	help='path to create data')
parser.add_argument('--result_path', dest='result_path', type=str, default='/data/', 
	help='path to create data')
parser.add_argument('--sampling_scheme', dest='sampling_scheme', type=str, 
	default='random', help='sampling scheme')
parser.add_argument('--num_slices', dest='num_slices', type=int, default=10, 
	help='number of slices to show')
parser.add_argument('--input_format', dest='input_format', type=str, default='hdf5', 
	help='mat or hdf5')
args = parser.parse_args()

if __name__ == '__main__':
    plotting_class = plotting_class(args)
    plotting_class.plot_sr_result()