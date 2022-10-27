import copy
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

# --- Local imports
from models.autoencoder import *

# --- Load dataset
# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # convert data to torch.FloatTensor
# transform = transforms.ToTensor()

# # load the training and test datasets
# train_data = datasets.MNIST(root='./tmp/', train=True, download=True, transform=transform)
# test_data = datasets.MNIST(root='./tmp/', train=False, download=True, transform=transform)

# # Create training and test dataloaders
# num_workers = 0
# # how many samples per batch to load
batch_size = 1

# # prepare data loaders
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    
# # obtain one batch of training images
# dataiter = iter(train_loader)
# images, labels = dataiter._next_data()
# images = images.numpy()

# # get one image from the batch
# img = np.squeeze(images[0])

# plt.imshow(img, cmap='gray')
# plt.show()

# --- Create sine wave
fs = 30000
f = 500
samples = 256
x = np.sin(2 * np.pi * f * np.arange(samples) / fs)

# --- Network and loss definition
# initialize the NN
model = ConvAutoencoder()
print(model)

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --- Training
# number of epochs to train the model
n_epochs = 150

# Prepare data for training
x = torch.from_numpy(x).float()
x = torch.unsqueeze(x, 0)
x = torch.unsqueeze(x, 0) # Expanding to (B, C, L) format (B=batch size, C=channels, L=length)
x.requires_grad = True

best_loss = 99999.0

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    # for data in train_loader:
        # _ stands in for labels, here
        # no need to flatten images
    
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    # forward pass: compute predicted outputs by passing inputs to the model
    outputs = model(x)
    # calculate the loss
    loss = criterion(outputs, x)
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()
    # update running training loss
    train_loss += loss.item()*len(x)
            
    # print avg training statistics 
    train_loss = train_loss
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
        epoch, 
        train_loss
        ))

    if train_loss < best_loss:
        best_model = copy.deepcopy(model)

# --- Inference
best_model.eval()

with torch.no_grad():
    best_model.training = False
    x_hat = best_model(x)

# x = x.view(16, 1, input_dim)
plt.plot(x.cpu().detach().numpy()[0][0])
plt.plot(x_hat.cpu()[0][0])
plt.show()

# # obtain one batch of test images
# data = x

# # get sample outputs
# output = model(images)
# # prep images for display

# # use detach when it's an output that requires_grad
# output = output.detach().numpy()

# # plot the first ten input images and then reconstructed images
# plt.plot(data.cpu().detach().numpy()[0][0])
# plt.plot(output)
# plt.show()