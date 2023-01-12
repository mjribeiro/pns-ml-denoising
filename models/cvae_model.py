import math
import torch
import torch.nn as nn
import torch.nn.functional as F

no_cuda = False
is_cuda = not no_cuda and torch.cuda.is_available()


# Source: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    if is_cuda:
        U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(latent_dim, categorical_dim, logits, temperature, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y.view(-1, latent_dim, categorical_dim)

    y_shape = y.size()
    _, ind = y.max(dim=-1)

    # Create array for onehot/hard vector?
    y_hard = torch.zeros_like(y).view(-1, y_shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*y_shape)
    
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y

    return y_hard.view(-1, latent_dim, categorical_dim)


# Based on: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
class CoordinateEncoder(nn.Module):
    def __init__(self, input_dim, conv_out_dim, output_dim, kernel_size, num_layers, pool_step, device):
        super(CoordinateEncoder, self).__init__()

        self.device = device

        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.pool_step   = pool_step
        self.num_layers  = num_layers
        self.layers      = nn.ModuleList()

        in_channels = [input_dim, 18, 36]
        out_channels = [18, 36, conv_out_dim]

        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(in_channels=in_channels[i], 
                                         out_channels=out_channels[i], 
                                         kernel_size=kernel_size, 
                                         padding=math.floor(kernel_size/2), 
                                         stride=1))

        self.maxpool    = nn.MaxPool1d(kernel_size=self.pool_step, stride=self.pool_step)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout    = nn.Dropout(0.05)
        self.fcn        = nn.Linear(16, output_dim)


    def forward(self, x):
        h_ = x

        for layer in self.layers:
            h_   = self.dropout(self.leaky_relu(self.maxpool(layer(h_))))

        # Add final dense layer
        return self.fcn(h_)


class Encoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim, kernel_size, num_layers, pool_step, batch_size, device):
        super(Encoder, self).__init__()

        self.device = device

        self.input_dim   = input_dim
        self.latent_dim  = latent_dim
        self.pool_step   = pool_step
        self.num_layers  = num_layers
        self.layers      = nn.ModuleList()

        # WIP
        for i in range(self.num_layers-1):
            self.layers.append(nn.Conv1d(in_channels=input_dim * (2 ** i), 
                                         out_channels=input_dim * (2 ** (i + 1)), 
                                         kernel_size=kernel_size, 
                                         padding=math.floor(kernel_size/2), 
                                         stride=1))

        # Final convolutional layer feeding into latent space
        self.layers.append(nn.Conv1d(in_channels=input_dim * (2 ** (i + 1)), 
                                     out_channels=latent_dim, 
                                     kernel_size=kernel_size, 
                                     padding=math.floor(kernel_size/2), 
                                     stride=1))

        self.maxpool    = nn.MaxPool1d(kernel_size=self.pool_step, stride=self.pool_step, return_indices=True)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout    = nn.Dropout(0.05)

    
    def forward(self, x):
        h_ = x
        maxpool_indices = []
        for layer in self.layers:
            # Check if dimension of data would be reduced to less than the size of convolutional layer
            if (h_.shape[2] / self.pool_step) < h_.shape[1]:
                h_ = layer(h_)
            else:
                h_, indices   = self.maxpool(layer(h_))
                maxpool_indices.append(indices)

            h_            = self.dropout(self.leaky_relu(h_))

        return h_, maxpool_indices


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, kernel_size, num_layers, pool_step, device):
        super(Decoder, self).__init__()
        
        self.device = device
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.pool_step = pool_step
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # TODO: Write this in a more compact way
        if pool_step == 8: maxpool_layer_count = 1
        elif pool_step == 4: maxpool_layer_count = 2
        elif pool_step == 2: maxpool_layer_count = 3

        for i in range(self.num_layers - maxpool_layer_count):
            self.layers.append(nn.ConvTranspose1d(in_channels=latent_dim * (2 ** i), 
                                                  out_channels=latent_dim * (2 ** (i + 1)), 
                                                  kernel_size=kernel_size,
                                                  padding=math.floor(kernel_size/2), 
                                                  stride=1))
        last_i = i + 1

        for i in range(maxpool_layer_count-1):
            self.layers.append(nn.ConvTranspose1d(in_channels=latent_dim * (2 ** last_i), 
                                                  out_channels=latent_dim * (2 ** (last_i + 1)), 
                                                  kernel_size=self.pool_step,
                                                  padding=0, 
                                                  stride=self.pool_step))
            last_i += 1

        self.layers.append(nn.ConvTranspose1d(in_channels=latent_dim * (2 ** last_i), 
                                              out_channels=output_dim, 
                                              kernel_size=self.pool_step,
                                              padding=0, 
                                              stride=self.pool_step))

        # WIP
        self.maxunpool  = nn.MaxUnpool1d(kernel_size=self.pool_step, stride=self.pool_step)
        self.dropout    = nn.Dropout(0.05)
        self.leaky_relu = nn.LeakyReLU()
        self.tanh       = nn.Tanh()

        # TODO: See if these fully connected layers are needed to getting right input shape again
        # self.fc_up = nn.Linear(latent_dim, int(output_dim/(2**num_layers))*latent_dim)
        

    def forward(self, x, maxpool_indices):
        h = x
        maxpool_indices = list(reversed(maxpool_indices))

        for layer_idx in range(self.num_layers-1):
            h   = self.leaky_relu(self.dropout(self.layers[layer_idx](h)))
            # h = self.leaky_relu(self.dropout(self.layers[layer_idx](self.maxunpool(h, maxpool_indices[layer_idx]))))
        return self.tanh(self.dropout(self.layers[-1](h)))
    

class CoordinateVAEModel(nn.Module):
    def __init__(self, encoder, decoder, coordinate_encoder):
        super(CoordinateVAEModel, self).__init__()
        # self.CoordinateEncoder = CoordinateEncoder
        self.encoder = encoder
        self.decoder = decoder
        self.coordinate_encoder = coordinate_encoder

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()

        self.epoch = 0
        self.tau = 0.1
        self.training = True
    
    
    def forward(self, x):
        # Get logits from encoder
        q_y, maxpool_indices = self.encoder(x)
        coord_encoding = self.coordinate_encoder(x)

        # Gumbel-Softmax activation on the latent space
        self.tau = 2*math.exp(-0.0003*self.epoch)

        if self.training == False:
            self.tau = 0.1

        # Sample from gumbel distribution and return onehot
        # z = gumbel_softmax(latent_dim, categorical_dim, q_y, self.tau, hard=True)
        z = F.gumbel_softmax(q_y, tau=self.tau, hard=True)

        # Concatenate sample of coordinate encoding
        p = torch.tensor(torch.ones(coord_encoding.shape[1]) / coord_encoding.shape[1])
        idx = p.multinomial(num_samples=10, replacement=False)
        z_coord = coord_encoding[:, idx, :]

        z = torch.concat((z, z_coord), dim=1)
        
        # Get output from decoder
        # return self.Decoder(z), F.softmax(q_y, dim=-1).reshape(*q.size()), categorical_dim
        return self.decoder(z, maxpool_indices)