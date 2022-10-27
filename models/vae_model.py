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

        in_channels = [2, 32]
        out_channels = [32, 32]
        
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(in_channels=in_channels[i], 
                                         out_channels=out_channels[i], 
                                         kernel_size=kernel_size, 
                                         padding=math.floor(kernel_size/2), 
                                         stride=1))

        self.maxpool    = nn.MaxPool1d(kernel_size=self.pool_step, stride=self.pool_step)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout    = nn.Dropout(0.05)

        self.fc_mu = nn.Linear(32*32, 30)
        self.fc_logvar = nn.Linear(32*32, 30)

    
    def forward(self, x):
        batch, _, _ = x.shape
        h_ = x
        for layer in self.layers:
            h_   = self.dropout(self.leaky_relu(self.maxpool(layer(h_))))

        h_latent = h_.reshape(batch, -1)

        mu = self.fc_mu(h_latent)
        logvar = self.fc_logvar(h_latent)

        # I think this sort of "flattening" helps a bit with the onehot?
        # h_ = h_.view(h_.shape[0], latent_size * data_length)
        
        return h_latent, mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, kernel_size, num_layers, pool_step, device):
        super(Decoder, self).__init__()
        
        self.device = device
        
        self.pool_step = pool_step
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        in_channels = [30, 32]
        out_channels = [32, 2]

        for i in range(self.num_layers):
            self.layers.append(nn.ConvTranspose1d(in_channels=in_channels[i], 
                                                  out_channels=out_channels[i], 
                                                  kernel_size=self.pool_step,
                                                  padding=0, 
                                                  stride=self.pool_step))

        self.dropout    = nn.Dropout(0.05)
        self.leaky_relu = nn.LeakyReLU()
        self.tanh       = nn.Tanh()

        # TODO: See if these fully connected layers are needed to getting right input shape again
        self.fc_up = nn.Linear(30, 32*30)
        

    def forward(self, x):
        h = x
        
        # FC layer to get original downsampled dims
        h = self.fc_up(h)

        # Reshape for conv1d
        h = h.reshape(-1, 30, 32)

        for layer_idx in range(self.num_layers-1):
            h   = self.leaky_relu(self.dropout(self.layers[layer_idx](h)))
        return self.tanh(self.dropout(self.layers[-1](h)))
    

class CoordinateVAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(CoordinateVAEModel, self).__init__()
        # self.CoordinateEncoder = CoordinateEncoder
        self.Encoder = Encoder
        self.Decoder = Decoder

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

        self.epoch = 0
        self.tau = 0.1
        self.training = True


    # Source: https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
    def sampler(self, mu, logvar):
        sigma = torch.exp(logvar)
        return mu + sigma * self.N.sample(mu.shape)
    
    
    def forward(self, x):
        # Get logits from encoder
        latent_data, mu, logvar = self.Encoder(x)

        z = self.sampler(mu, logvar)

        # # Gumbel-Softmax activation on the latent space
        # self.tau = 2*math.exp(-0.0003*self.epoch)

        if self.training == True:
            y_hard = F.gumbel_softmax(z, tau=self.tau, hard=True)
            y_soft = F.gumbel_softmax(z, tau=self.tau, hard=False)
            z = y_hard - y_soft.detach() + y_soft
            # latent_data = F.gumbel_softmax(logits, tau=tau, hard=True)
        else:
            y_hard = F.gumbel_softmax(z, tau=0.1, hard=True)
            y_soft = F.gumbel_softmax(z, tau=0.1, hard=False)
            z = y_hard - y_soft.detach() + y_soft
            # latent_data = F.gumbel_softmax(logits, tau=0.1, hard=True)
        
        # See comment in encoder about flattening
        # latent_data = latent_data.view(onehot.shape[0], latent_size, data_length)
        
        # Get output from decoder
        if self.training == True:
            return self.Decoder(z)
        else:
            return self.Decoder(z), z
