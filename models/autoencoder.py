import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 16), 3x3 kernels
        self.conv1 = nn.Conv1d(1, 32, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv1d(32, 12, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool1d(2, 2)
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose1d(12, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(32, 1, 2, stride=2)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = self.dropout(F.leaky_relu(self.conv1(x), 0.2))
        x = self.pool(x)
        # add second hidden layer
        x = self.dropout(F.leaky_relu(self.conv2(x), 0.2))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.leaky_relu(self.dropout(self.t_conv1(x)), 0.2)
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.tanh(self.dropout(self.t_conv2(x)))
                
        return x
