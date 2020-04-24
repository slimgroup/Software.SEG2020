# This script is partly obtained from https://github.com/ZezhouCheng/GP-DIP

import torch
import torch.nn as nn
import numpy as np
from helper_function import *
from math import floor, ceil

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
torch.nn.Module.add = add_module

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

class Crop(nn.Module):
    def __init__(self, model_shape, d_dim=None):
        super(Crop, self).__init__()

        if d_dim == None:
            pwr0 = 1
            while pwr0 < np.array(model_shape[2]):
                pwr0 *= 2
            pwr1 = 1
            while pwr1 < np.array(model_shape[3]):
                pwr1 *= 2
            d_dim = [pwr0,pwr1]

        self.downsample_num = int(np.log(np.min(np.array(d_dim)))/np.log(2) - 2)
        self.padding_dim, self.desired_dim = find_padding_dim(input_dim=np.array(model_shape[2:]), 
            d_dim=d_dim)

    def forward(self, input):
        return input[:, :, self.padding_dim[0, 0]:-self.padding_dim[0, 1], 
                self.padding_dim[1, 0]:-self.padding_dim[1, 1]]


def find_padding_dim(input_dim, d_dim=[32, 32]):

    desired_dim = [d_dim[0], d_dim[1]]
    padding_dim = np.zeros([2, 2], dtype=int)
    padding_dim[0, 0] = int(floor((desired_dim[0] - input_dim[0])/2.))
    padding_dim[0, 1] = int(ceil((desired_dim[0] - input_dim[0])/2.))
    padding_dim[1, 0] = int(floor((desired_dim[1] - input_dim[1])/2.))
    padding_dim[1, 1] = int(ceil((desired_dim[1] - input_dim[1])/2.))
    return padding_dim, desired_dim

class generator(nn.Module):
    def __init__(self, model_shape,
        num_input_channels=1, num_output_channels=1, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], 
        num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3, filter_size_up=3, 
        filter_skip_size=1, need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
        super(generator, self).__init__()

        """Assembles encoder-decoder with skip connections.
        Arguments:
            act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
            pad (string): zero|reflection (default: 'zero')
            upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
            downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
        """
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        n_scales = len(num_channels_down) 

        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
            upsample_mode   = [upsample_mode]*n_scales

        if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
            downsample_mode   = [downsample_mode]*n_scales
        
        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
            filter_size_down   = [filter_size_down]*n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
            filter_size_up   = [filter_size_up]*n_scales

        last_scale = n_scales - 1 

        cur_depth = None

        self.model = nn.Sequential()
        model_tmp = self.model

        input_depth = num_input_channels
        for i in range(len(num_channels_down)):

            deeper = nn.Sequential()
            skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                model_tmp.add(Concat(1, skip, deeper))
            else:
                model_tmp.add(deeper)
            
            model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

            if num_channels_skip[i] != 0:
                skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
                skip.add(bn(num_channels_skip[i]))
                skip.add(act(act_fun))
                
            # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

            deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))

            deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
            deeper.add(bn(num_channels_down[i]))
            deeper.add(act(act_fun))

            deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
                # The deepest
                k = num_channels_down[i]
            else:
                deeper.add(deeper_main)
                k = num_channels_up[i + 1]

            if not upsample_mode[i] == 'nearest':
                deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=True))
            else:
                deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))                

            model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))


            if need1x1_up:
                model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
                model_tmp.add(bn(num_channels_up[i]))
                model_tmp.add(act(act_fun))

            input_depth = num_channels_down[i]
            model_tmp = deeper_main

        self.model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
        if need_sigmoid:
            self.model.add(nn.Sigmoid())

        # self.model.add(Crop(model_shape, d_dim=[1024, 640]))
        self.model.add(Crop(model_shape, d_dim=[256, 192]))

    # def forward(self, input):
    #     output = self.model(input)*torch.Tensor([1.0, 0.1, 0.1]).unsqueeze_(1).unsqueeze_(1).unsqueeze_(0)
    #     return output*0.5

    def forward(self, input):
        return self.model(input)/40.0