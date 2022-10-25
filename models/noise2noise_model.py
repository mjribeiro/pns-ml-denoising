import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class Noise2NoiseEncoder(nn.Module):
    def __init__(self, pool_step, kernel_size, num_layers) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        in_channels = []
        out_channels = []
        
        for i in range(self.num_layers-1):
            self.layers.append(nn.Conv1d(in_channels=in_channels[i], 
                                         out_channels=out_channels[i], 
                                         kernel_size=kernel_size, 
                                         padding=math.floor(kernel_size/2), 
                                         stride=1))
        self.layers.append(nn.Conv1d(in_channels=in_channels[-1], 
                                     out_channels=out_channels[-1], 
                                     kernel_size=kernel_size, 
                                     padding=math.floor(kernel_size/2), 
                                     stride=1))
        
        self.maxpool    = nn.MaxPool1d(kernel_size=pool_step, stride=pool_step)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        h_   = self.maxpool(self.leaky_relu(self.layers[0](x)))
        for layer in self.layers:
            h_   = self.maxpool(self.leaky_relu(layer(h_)))


class Noise2NoiseDecoder(nn.Module):
    def __init__(self, pool_step) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        in_channels = []
        out_channels = []

        for i in range(self.num_layers-1):
            self.layers.append(nn.ConvTranspose1d(in_channels=in_channels[i], 
                                                  out_channels=out_channels[i], 
                                                  kernel_size=pool_step,
                                                  padding=0, 
                                                  stride=pool_step))
        self.layers.append(nn.ConvTranspose1d(in_channels=in_channels[-1], 
                                              out_channels=out_channels[-1],
                                              kernel_size=pool_step, 
                                              padding=0, 
                                              stride=pool_step))

    def forward(self, x):
        pass

class Noise2NoiseModel(nn.Module):
    def __init__(self):
        super(Noise2NoiseModel).__init__()

    def forward(self, x):
        pass
