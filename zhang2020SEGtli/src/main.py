import argparse
import os
import torch
import numpy as np
from model import wavefield_reconstruction
np.random.seed(19)
torch.manual_seed(19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment', dest='experiment', default='interpolation', 
    help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=300, 
    help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=150, 
    help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, 
    help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, 
    help='initial learning rate for adam')
parser.add_argument('--phase', dest='phase', default='train', 
    help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=10000, 
    help='save a model every save_freq iterations')
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=100, 
    help='sample_freq')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', 
    help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', 
    help='sample are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./log', 
    help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', 
    help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=1000.0, 
    help='weight on L1 term in objective')
parser.add_argument('--data_path', dest='data_path', type=str, default='/home/ec2-user/data/', 
    help='path of the train/test dataset')
parser.add_argument('--freq', dest='frequency', type=float, default=14.33, 
    help='frequency of data')
parser.add_argument('--sampling_rate', dest='sampling_rate', type=float, default=0.1, 
    help='sampling rate')
parser.add_argument('--sampling_scheme', dest='sampling_scheme', type=str, default='random', 
    help='sampling scheme')
parser.add_argument('--cuda', dest='cuda', type=int, default=1, help='set it to 1 for running on GPU, 0 for CPU')
args = parser.parse_args()


def main():
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(os.path.join(args.checkpoint_dir, args.experiment)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.experiment))
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: Cuda is not available, try running on CPU")
        sys.exit(1)

    model = wavefield_reconstruction(args)
    model.train(args) if args.phase == 'train' \
        else model.test(args)

if __name__ == '__main__':
    main()
