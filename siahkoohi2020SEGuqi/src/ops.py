import torch
import torch.nn as nn
import copy
import numpy as np
from math import floor, ceil
torch.manual_seed(19)

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=True):
        super(ConvLayer, self).__init__()
        self.padding = padding
        padding_0 = (kernel_size - 1) // 2
        padding_1 = (kernel_size - 1) - padding_0
        self.reflection_pad = torch.nn.ReflectionPad2d((padding_0, padding_1, padding_0, padding_1))
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.padding:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, mode='bilinear'):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.mode = mode
        padding_0 = (kernel_size - 1) // 2
        padding_1 = (kernel_size - 1) - padding_0
        self.reflection_pad = torch.nn.ReflectionPad2d((padding_0, padding_1, padding_0, padding_1))
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode=self.mode, scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class Deconv2D(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(Deconv2D, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv2d = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        # out = self.reflection_pad(self.reflection_pad(x))
        out = self.conv2d(x)
        return out


def weights_init_impulse(paramList):

    for param in list(paramList):
        if param.dim() == 4:
            param.data[:, :, param.size(2)//2, param.size(3)//2] = 1.0

def weights_init_costume(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def weights_init_xavier_normal(paramList):
    for param in list(paramList):
        if param.dim() == 4:
            torch.nn.init.xavier_normal_(param)


def find_padding_dim(input_dim, downsample_num=4):

    factor = 2**downsample_num

    desired_dim = copy.copy(input_dim)
    if np.mod(input_dim[0], factor) > 0:
        desired_dim[0] = input_dim[0] + factor - np.mod(input_dim[0], factor)
    if np.mod(input_dim[1], factor) > 0:
        desired_dim[1] = input_dim[1] + factor - np.mod(input_dim[1], factor)

    padding_dim = np.zeros([2, 2], dtype=int)
    padding_dim[0, 0] = int(floor((desired_dim[0] - input_dim[0])/2.))
    padding_dim[0, 1] = int(ceil((desired_dim[0] - input_dim[0])/2.))

    padding_dim[1, 0] = int(floor((desired_dim[1] - input_dim[1])/2.))
    padding_dim[1, 1] = int(ceil((desired_dim[1] - input_dim[1])/2.))

    return padding_dim, desired_dim

