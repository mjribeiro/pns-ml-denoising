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
    
    def __init__(self, input_dim, hidden_dim, kernel_size, latent_dim):
        super(Encoder, self).__init__()

        self.conv1d_1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=1)
        self.conv1d_2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=kernel_size, stride=1)
        self.conv1d_3 = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=latent_dim, kernel_size=kernel_size, stride=1)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)
        
        self.training = True
    
    
    def forward(self, x):
        h_   = self.dropout(self.leaky_relu(self.maxpool(self.conv1d_1(x))))
        h_   = self.dropout(self.leaky_relu(self.maxpool(self.conv1d_2(h_))))
        return self.dropout(self.leaky_relu(self.maxpool(self.conv1d_3(h_))))


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, kernel_size):
        super(Decoder, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=latent_dim, out_channels=hidden_dim, kernel_size=kernel_size, stride=1)
        self.conv1d_2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=kernel_size, stride=1)
        self.conv1d_3 = nn.Conv1d(in_channels=hidden_dim * 2, out_channels=output_dim, kernel_size=kernel_size, stride=1)
        
        self.upsampling = nn.Upsample(scale_factor=2)
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu  = nn.LeakyReLU(0.2)
        
        

    def forward(self, x):
        h    = self.leaky_relu(self.conv1d(self.dropout(self.upsampling(x))))
        h    = self.leaky_relu(self.conv1d(self.dropout(self.upsampling(h))))
        return self.leaky_relu(self.conv1d(self.dropout(self.upsampling(h))))
    

class CoordinateVAEModel(nn.Module):
    def __init__(self, Encoder, Decoder, device):
        super(CoordinateVAEModel, self).__init__()
        # self.CoordinateEncoder = CoordinateEncoder
        self.Encoder = Encoder
        self.Decoder = Decoder

        self.epoch = 0

    def forward(self, x):
        # Get logits from encoder
        logits = self.Encoder(x)

        # Gumbel-Softmax activation on the latent space
        tau = 2 ** (-0.0003*self.epoch)

        if self.Encoder.training == True:
            onehot = F.gumbel_softmax(logits, tau=0.1, hard=True)
        else:
            onehot = F.gumbel_softmax(logits, tau=tau, hard=True)

        # Get output from decoder

        return 0
