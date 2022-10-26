import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Implement skips and concatenation

class Noise2NoiseEncoder(nn.Module):
    def __init__(self, num_layers=7, kernel_size=3, pool_step=2) -> None:
        super(Noise2NoiseEncoder, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        in_channels = [1, 48, 48, 48, 48, 48, 48]
        out_channels = [48, 48, 48, 48, 48, 48, 48]
        
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

        # Layer 1, just conv_lr
        h_ = self.leaky_relu(self.layers[0](x))

        # Layer 2, conv_lr and maxpool
        encodings.append(self.maxpool(self.leaky_relu(self.layers[1](h_))))

        # Layers 3-5, conv_lr and maxpool
        for layer_idx in range(2, self.num_layers-2):
            encodings.append(self.maxpool(self.leaky_relu(self.layers[layer_idx](encodings[-1]))))

        # Layer 6, conv_lr and maxpool
        h_ = self.maxpool(self.leaky_relu(self.layers[-2](encodings[-1])))

        # Layer 7, conv_lr only
        return self.leaky_relu(self.layers[-1](h_)), encodings


class Noise2NoiseDecoder(nn.Module):
    def __init__(self, data_length, num_layers=11, kernel_size=3, pool_step=2) -> None:
        super(Noise2NoiseDecoder, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        in_channels = [48, 96, 96, 96, 96, 96, 96, 96, 96, 64, 32]
        out_channels = [96, 96, 96, 96, 96, 96, 96, 96, 64, 32, 1]

        for i in range(0, num_layers-1, 2):
            self.layers.append(nn.ConvTranspose1d(in_channels=in_channels[i], 
                                                  out_channels=out_channels[i], 
                                                  kernel_size=pool_step,
                                                  padding=0, 
                                                  stride=pool_step))
            self.layers.append(nn.Conv1d(in_channels=in_channels[i + 1], 
                                         out_channels=out_channels[i + 1], 
                                         kernel_size=kernel_size, 
                                         padding=math.floor(kernel_size/2), 
                                         stride=1))
        self.layers.append(nn.Conv1d(in_channels=in_channels[-1], 
                                         out_channels=out_channels[-1], 
                                         kernel_size=kernel_size, 
                                         padding=math.floor(kernel_size/2), 
                                         stride=1))

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.tanh = nn.Tanh()
        # self.linear = nn.Linear(in_features=data_length,
        #                         out_features=data_length)

    def forward(self, x, encodings):
        # Reverse encodings list, later encoding layers go in first
        encodings = list(reversed(encodings))

        h = x
        for layer_idx in range(0, self.num_layers-1, 2):
            h   = self.leaky_relu(self.layers[layer_idx](h))
            h   = torch.cat((h, encodings[int(layer_idx/2)]), dim=1)
            h   = self.leaky_relu(self.layers[layer_idx + 1](h))
        return self.tanh(self.layers[-1](h))


class Noise2NoiseModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Noise2NoiseModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x):
        logits, encodings = self.encoder(x)
        return self.decoder(logits, encodings)
