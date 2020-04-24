import argparse
import os
import sys
import torch
import numpy as np
from model import LearnedImaging
from sample import Sample
np.random.seed(19)
torch.manual_seed(19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=1001, 
    help='# of epoch')
parser.add_argument('--inner_iter', dest='inner_iter', type=int, default=10, 
    help='# of inner_iter')
parser.add_argument('--eta', dest='eta', type=float, default=0.01, 
    help='noise_amp')
parser.add_argument('--batchsize', dest='batchsize', type=int, default=1, 
    help='batchsize')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, 
    help='initial learning rate for adam')
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0., 
    help='weight_decay')
parser.add_argument('--phase', dest='phase', default='train', 
    help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=100, 
    help='save a model every save_freq iterations')
parser.add_argument('--sample_freq', dest='sample_freq', type=int, default=1, 
    help='sample_freq')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', 
    help='models are saved here')
parser.add_argument('--experiment', dest='experiment', default='learned-imaging', 
    help='experiment name')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', 
    help='sample are saved here')
parser.add_argument('--cuda', dest='cuda', type=int, default=0, 
    help='set it to 1 for running on GPU, 0 for CPU')
parser.add_argument('--vel_dir', dest='vel_dir', default='./vel_dir', 
    help='path to save velocity model')
args = parser.parse_args()

def main():
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(os.path.join(args.checkpoint_dir, args.experiment)):
        os.makedirs(os.path.join(args.checkpoint_dir, args.experiment))
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(os.path.join(args.sample_dir, args.experiment)):
        os.makedirs(os.path.join(args.sample_dir, args.experiment))
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: Cuda is not available, try running on CPU")
        sys.exit(1)

    if args.phase == 'train':
        model = LearnedImaging(args)
        model.train(args)
    else:
        sample = Sample(args)
        if args.phase == 'test':
            sample.test(args)

if __name__ == '__main__':
    main()
