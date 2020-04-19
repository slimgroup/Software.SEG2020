import torch
import numpy as np
from ops import *

class generator(torch.nn.Module):
    def __init__(self, num_channels=64):
        super(generator, self).__init__()
        # Initial convolution layers

        self.conv1 = ConvLayer(2, num_channels, kernel_size=7, stride=1,  padding=False)
        self.in1 = torch.nn.InstanceNorm2d(num_channels, affine=True)
        self.conv2 = ConvLayer(num_channels, num_channels*2, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(num_channels*2, affine=True)
        self.conv3 = ConvLayer(num_channels*2, num_channels*4, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(num_channels*4, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(num_channels*4)
        self.res2 = ResidualBlock(num_channels*4)
        self.res3 = ResidualBlock(num_channels*4)
        self.res4 = ResidualBlock(num_channels*4)
        self.res5 = ResidualBlock(num_channels*4)
        self.res6 = ResidualBlock(num_channels*4)
        self.res7 = ResidualBlock(num_channels*4)
        self.res8 = ResidualBlock(num_channels*4)
        self.res9 = ResidualBlock(num_channels*4)
        # Upsampling Layers
        self.deconv1 = Deconv2D(num_channels*4, num_channels*2, kernel_size=3, stride=2)
        self.in4 = torch.nn.InstanceNorm2d(num_channels*2, affine=True)
        self.deconv2 = Deconv2D(num_channels*2, num_channels, kernel_size=3, stride=2)
        self.in5 = torch.nn.InstanceNorm2d(num_channels, affine=True)
        self.deconv3 = ConvLayer(num_channels, 2, kernel_size=7, stride=1, padding=False)

        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        y = torch.nn.ReflectionPad2d((5, 5, 5, 5))(x)
        y = self.relu(self.in1(self.conv1(y)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)
        y = self.res8(y)
        y = self.res9(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = torch.nn.ReflectionPad2d((3, 2, 3, 2))(y)
        y = self.deconv3(y)

        return y

class discriminator(torch.nn.Module):
    def __init__(self, num_channels=64):
        super(discriminator, self).__init__()
        # Initial convolution layers

        self.conv1 = ConvLayer(2, num_channels, kernel_size=4, stride=2,  padding=False)
        
        self.conv2 = ConvLayer(num_channels, num_channels*2, kernel_size=4, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(num_channels*2, affine=True)
        
        self.conv3 = ConvLayer(num_channels*2, num_channels*4, kernel_size=4, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(num_channels*4, affine=True)
        
        self.conv4 = ConvLayer(num_channels*4, num_channels*8, kernel_size=4, stride=1)
        self.in4 = torch.nn.InstanceNorm2d(num_channels*8, affine=True)
        
        self.conv5 = ConvLayer(num_channels*8, 1, kernel_size=1, stride=1)
        # Non-linearities
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x):
        y = self.lrelu(self.conv1(x))
        y = self.lrelu(self.in2(self.conv2(y)))
        y = self.lrelu(self.in3(self.conv3(y)))
        y = self.lrelu(self.in4(self.conv4(y)))
        y = self.conv5(y)
        return y


