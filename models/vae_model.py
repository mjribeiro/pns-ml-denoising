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

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim, categorical_dim)


# Based on: https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
class CoordinateEncoder(nn.Module):
    def __init__(self):
        super(CoordinateEncoder, self).__init__()


    def forward(self, x):
        pass


class Encoder(nn.Module):
    
    def __init__(self, input_dim, latent_dim, kernel_size, num_layers, pool_step, batch_size, device):
        super(Encoder, self).__init__()

        self.device = device

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pool_step = pool_step
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        in_channels = [2, 64, 128]
        out_channels = [64, 128, latent_dim]
        
        for i in range(self.num_layers):
            self.layers.append(nn.Conv1d(in_channels=in_channels[i], 
                                         out_channels=out_channels[i], 
                                         kernel_size=kernel_size, 
                                         padding=math.floor(kernel_size/2), 
                                         stride=1))

        self.maxpool    = nn.MaxPool1d(kernel_size=self.pool_step, stride=self.pool_step)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout    = nn.Dropout(0.05)

        # self.fc_mu = nn.Linear(batch_size*out_channels[-1]*int(input_dim/(2**num_layers)), latent_dim)
        # self.fc_logvar = nn.Linear(batch_size*out_channels[-1]*int(input_dim/(2**num_layers)), latent_dim)

    
    def forward(self, x):
        h_ = x
        for layer in self.layers:
            h_   = self.dropout(self.leaky_relu(self.maxpool(layer(h_))))

        q = h_.view(h_.shape[0], self.latent_dim*h_.shape[2])
        
        # mu = self.fc_mu(h_latent)
        # logvar = self.fc_logvar(h_latent)

        # I think this sort of "flattening" helps a bit with the onehot?
        # h_ = h_.view(h_.shape[0], latent_size * data_length)
        
        # return h_latent, mu, logvar
        return q, h_, self.latent_dim, h_.shape[2]


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, kernel_size, num_layers, pool_step, device):
        super(Decoder, self).__init__()
        
        self.device = device
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.pool_step = pool_step
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        in_channels = [latent_dim, 128, 64]
        out_channels = [128, 64, 2]

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
        # self.fc_up = nn.Linear(latent_dim, int(output_dim/(2**num_layers))*latent_dim)
        

    def forward(self, x):
        h = x
        
        # FC layer to get original downsampled dims
        # h = self.fc_up(h)

        # Reshape for conv1d
        # h = h.reshape(-1, self.latent_dim, int(self.output_dim/(2**self.num_layers)))

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
        q, q_y, latent_dim, categorical_dim = self.Encoder(x)

        # Gumbel-Softmax activation on the latent space
        self.tau = 2*math.exp(-0.0003*self.epoch)

        # z = self.sampler(mu, logvar)
        # q_y = q.view(q.shape[0], latent_dim, categorical_dim)
        z = gumbel_softmax(latent_dim, categorical_dim, q_y, self.tau, hard=True)

        # if self.training == True:
        #     y_hard = F.gumbel_softmax(z, tau=self.tau, hard=True)
        #     y_soft = F.gumbel_softmax(z, tau=self.tau, hard=False)
        #     z = y_hard - y_soft.detach() + y_soft
        #     # latent_data = F.gumbel_softmax(logits, tau=tau, hard=True)
        # else:
        #     y_hard = F.gumbel_softmax(z, tau=0.1, hard=True)
        #     y_soft = F.gumbel_softmax(z, tau=0.1, hard=False)
        #     z = y_hard - y_soft.detach() + y_soft
        #     # latent_data = F.gumbel_softmax(logits, tau=0.1, hard=True)
        
        # See comment in encoder about flattening
        # latent_data = latent_data.view(onehot.shape[0], latent_size, data_length)
        
        # Get output from decoder
        return self.Decoder(z), F.softmax(q_y, dim=-1).reshape(*q.size()), categorical_dim
