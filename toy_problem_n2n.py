import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torchsummary import summary

# Local imports
from models.noise2noise_model import *

# --- Create sine wave
fs = 30000
f = 10
samples = 2048
num_repeats = 1

x = np.zeros((num_repeats, samples))

for i in range(num_repeats):
    x[i, :] = np.sin(2 * np.pi * f * np.arange(samples) / fs)


x_flat = x.flatten()

# Take smaller windows from main sine signal
samples = 512
x_windows = []
for i in range(0, len(x_flat) - samples, samples):
    x_windows.append(x_flat[i:i+samples])

# Store in dataloaders
train_dataloader = DataLoader(x_windows, batch_size=8, shuffle=True)
test_dataloader = DataLoader(x_windows, batch_size=1, shuffle=False)

# --- Set up model
# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Weights&Biases initialisation
wandb.init(project="PNS Denoising",
           config = {
              "learning_rate": 0.0005,
              "epochs": 1000,
              "batch_size": 1,
              "kernel_size": 3,
              "mse_weight": 1})

config = wandb.config

print("Setting up Noise2Noise model...")

encoder = Noise2NoiseEncoder()
decoder = Noise2NoiseDecoder(data_length=len(x_flat))
model = Noise2NoiseModel(encoder=encoder, decoder=decoder).to(device)

# summary(model, [(1, samples)])

# --- Hyperparameter setup
mse_loss = nn.MSELoss()
kld_loss = nn.KLDivLoss()

def loss_function(x, x_hat, mse_weight=0.25):
    kld_weight = 1 - mse_weight

    mse = mse_loss(x, x_hat)
    kld = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    return (mse_weight * mse) + (kld_weight * -kld)


optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

# --- Training
print("Start training Noise2Noise model...")
model.train()

x_flat = torch.from_numpy(x_flat).to(device).float()
x_flat = torch.unsqueeze(x_flat, 0) # Expanding to (B, C, L)
x_flat.requires_grad = True

# Set up toy problem for PyTorch
if torch.cuda.is_available():
    model.cuda()

best_loss = 99999.0

for epoch in range(config.epochs):
    overall_loss = 0
    model.epoch = epoch + 1

    # for batch_idx, data in enumerate(train_dataloader):
    optimizer.zero_grad()

    x_hat = model(x_flat)
    loss = loss_function(x_flat, x_hat, mse_weight=config.mse_weight)
    
    overall_loss += loss.item()*len(x_flat)
    
    loss.backward()

    # # Uncomment to print out model parameters (check if gradients are there)
    # for param in model.parameters():
    #     print(param.grad)
        
    optimizer.step()

    if loss < best_loss:
        best_model = copy.deepcopy(model)
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss)
    wandb.log({"loss": overall_loss})
        
print("Finished!")

# --- Inference
best_model.eval()

with torch.no_grad():
    x_hat = best_model(x_flat)

    # Convert to numpy and send to cpu
    x_hat = x_hat.cpu().numpy()

plt.plot(x_flat.detach().cpu().numpy()[0])
plt.plot(x_hat[0])
plt.show()