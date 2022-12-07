import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implement skips and concatenation

class Noise2NoiseEncoder(nn.Module):
    def __init__(self, num_channels, num_layers=7, kernel_size=3, pool_step=2) -> None:
        super(Noise2NoiseEncoder, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        in_channels = [num_channels, 48, 48, 48, 48, 48, 48]
        out_channels = [48, 48, 48, 48, 48, 48, 48]
        # in_channels = [num_channels, 16, 16, 16, 16, 16, 16]
        # out_channels = [16, 16, 16, 16, 16, 16, 16]
        
        for i in range(num_layers):
            self.layers.append(nn.Conv1d(in_channels=in_channels[i], 
                                         out_channels=out_channels[i], 
                                         kernel_size=kernel_size, 
                                         padding=math.floor(kernel_size/2), 
                                         stride=1))
        
        self.maxpool    = nn.MaxPool1d(kernel_size=pool_step, stride=pool_step)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        encodings = [x]

        # Layer 0, just conv_lr
        h_ = self.leaky_relu(self.layers[0](x))

        # Layers 1-5, conv_lr and maxpool
        for layer_idx in range(1, self.num_layers-1):
            h_ = self.maxpool(self.leaky_relu(self.layers[layer_idx](h_)))
            if layer_idx < (self.num_layers-2):
                encodings.append(h_)

        # Layer 6, conv_lr only
        return self.leaky_relu(self.layers[-1](h_)), encodings


class Noise2NoiseDecoder(nn.Module):
    def __init__(self, num_channels, data_length, num_layers=11, kernel_size=3, pool_step=2) -> None:
        super(Noise2NoiseDecoder, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        in_channels = [96, 96, 144, 96, 144, 96, 144, 96, 96+num_channels, 64, 32]
        out_channels = [96, 96, 96, 96, 96, 96, 96, 96, 64, 32, num_channels]
        # in_channels = [16, 16, 32, 16, 32, 16, 32, 16, 16+num_channels, 64, 32]
        # out_channels = [16, 16, 16, 16, 16, 16, 16, 16, 64, 32, num_channels]

        for i in range(0, num_layers):
            self.layers.append(nn.Conv1d(in_channels=in_channels[i], 
                                         out_channels=out_channels[i], 
                                         kernel_size=kernel_size, 
                                         padding=math.floor(kernel_size/2), 
                                         stride=1))

        self.upsample = nn.Upsample(scale_factor=pool_step)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()

    def forward(self, x, encodings):
        # Reverse encodings list, later encoding layers get concatenated first
        encodings = list(reversed(encodings))

        h = x
        for layer_idx in range(0, self.num_layers-1):
            if layer_idx % 2 == 0:
                # Concatenate encodings to even-index hidden layers
                h   = self.upsample(h)
                h   = torch.cat([h, encodings[int(layer_idx/2)]], dim=1)
                h   = self.leaky_relu(self.layers[layer_idx](h))
            else:
                # Do conv1d+leakyrelu for the remainder
                h   = self.leaky_relu(self.layers[layer_idx](h)) 


        return self.tanh(self.layers[-1](h))


class Noise2NoiseModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Noise2NoiseModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        logits, encodings = self.encoder(x)
        return self.decoder(logits, encodings)
