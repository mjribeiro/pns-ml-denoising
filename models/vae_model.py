import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Based on: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
class CoordinateEncoder(nn.Module):
    def __init__(self):
        super(CoordinateEncoder, self).__init__()


    def forward(self, x):
        pass

class Encoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim, kernel_size, num_layers, pool_step, device):
        super(Encoder, self).__init__()

        self.device = device

        self.pool_step = pool_step
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # for i in range(self.num_layers-1):
        #     self.layers.append(nn.Conv1d(in_channels=input_dim * (2 ** i), 
        #                                  out_channels=input_dim * (2 ** (i + 1)), 
        #                                  kernel_size=kernel_size, 
        #                                  padding=math.floor(kernel_size/2), 
        #                                  stride=1))
        self.layers.append(nn.Conv1d(in_channels=1, 
                                     out_channels=16, 
                                     kernel_size=kernel_size, 
                                     padding=math.floor(kernel_size/2), 
                                     stride=1))
        self.layers.append(nn.Conv1d(in_channels=16, 
                                     out_channels=32, 
                                     kernel_size=kernel_size, 
                                     padding=math.floor(kernel_size/2), 
                                     stride=1))
        self.layers.append(nn.Conv1d(in_channels=32, 
                                     out_channels=1, 
                                     kernel_size=kernel_size, 
                                     padding=math.floor(kernel_size/2), 
                                     stride=1))

        self.maxpool    = nn.MaxPool1d(kernel_size=self.pool_step, stride=self.pool_step)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout    = nn.Dropout(0.05)
        
        self.training = True
    
    
    def forward(self, x):
        h_ = x
        for layer in self.layers:
            h_   = self.dropout(self.leaky_relu(self.maxpool(layer(h_))))
        return h_
        # h_ = x
        # for layer in self.layers:
        #     h_   = F.relu(self.maxpool(layer(h_)))
        # return h_


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, kernel_size, num_layers, pool_step, device):
        super(Decoder, self).__init__()
        
        self.device = device
        
        self.pool_step = pool_step
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # for i in range(self.num_layers-1):
        #     self.layers.append(nn.ConvTranspose1d(in_channels=latent_dim * (2 ** i), 
        #                                           out_channels=latent_dim * (2 ** (i + 1)), 
        #                                           kernel_size=self.pool_step,
        #                                           padding=0, 
        #                                           stride=self.pool_step))
        self.layers.append(nn.ConvTranspose1d(in_channels=1, 
                                              out_channels=16,
                                              kernel_size=self.pool_step, 
                                              padding=0, 
                                              stride=self.pool_step))
        self.layers.append(nn.ConvTranspose1d(in_channels=16, 
                                              out_channels=32,
                                              kernel_size=self.pool_step, 
                                              padding=0, 
                                              stride=self.pool_step))
        self.layers.append(nn.ConvTranspose1d(in_channels=32, 
                                              out_channels=1,
                                              kernel_size=self.pool_step, 
                                              padding=0, 
                                              stride=self.pool_step))

        self.dropout    = nn.Dropout(0.05)
        self.leaky_relu = nn.LeakyReLU()
        self.tanh       = nn.Tanh()
        

    def forward(self, x):
        h = x
        for layer_idx in range(self.num_layers-1):
            h   = self.leaky_relu(self.dropout(self.layers[layer_idx](h)))
        return self.tanh(self.dropout(self.layers[-1](h)))
        # h = x
        # for layer_idx in range(self.num_layers-1):
        #     h   = F.relu(self.layers[layer_idx](h))
        # return self.tanh(self.layers[-1](h))
    

class CoordinateVAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(CoordinateVAEModel, self).__init__()
        # self.CoordinateEncoder = CoordinateEncoder
        self.Encoder = Encoder
        self.Decoder = Decoder

        self.epoch = 0

    def forward(self, x):
        # Get logits from encoder
        logits = self.Encoder(x)

        # # Gumbel-Softmax activation on the latent space
        # tau = 2*math.exp(-0.0003*self.epoch)

        # if self.Encoder.training == True:
        #     y_hard = F.gumbel_softmax(logits, tau=tau, hard=True)
        #     y_soft = F.gumbel_softmax(logits, tau=tau, hard=False)
        #     onehot = y_hard - y_soft.detach() + y_soft
        #     # onehot = F.gumbel_softmax(logits, tau=tau, hard=True)
        # else:
        #     y_hard = F.gumbel_softmax(logits, tau=0.1, hard=True)
        #     y_soft = F.gumbel_softmax(logits, tau=0.1, hard=False)
        #     onehot = y_hard - y_soft.detach() + y_soft
        #     # onehot = F.gumbel_softmax(logits, tau=0.1, hard=True)

        # # Get output from decoder
        # return self.Decoder(onehot)
        return self.Decoder(logits)
