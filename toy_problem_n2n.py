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
f = 50
total_samples = 5 * 60 * fs # 5 mins
samples = 2048
num_repeats = 1

# x = np.sin(2 * np.pi * f * np.arange(total_samples) / fs)
x_clean = np.sin(2 * np.pi * f * np.arange(total_samples) / fs)
x_noisy = x_clean + np.random.normal(0, 0.6, total_samples)
x_medium = x_clean + np.random.normal(0, 0.3, total_samples)

# Also normalise clean
x_clean = 2 * (x_clean - np.min(x_medium)) / (np.max(x_medium) - np.min(x_medium)) - 1
# Normalise to [-1, 1]
x_noisy = 2 * (x_noisy - np.min(x_noisy)) / (np.max(x_noisy) - np.min(x_noisy)) - 1
x_medium = 2 * (x_medium - np.min(x_medium)) / (np.max(x_medium) - np.min(x_medium)) - 1


x_windows = []
for i in range(0, len(x_noisy) - samples, samples):
    x_windows.append((x_noisy[i:i+samples], x_medium[i:i+samples]))
# x_windows = np.asarray(x_windows)

# --- Set up model
# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Weights&Biases initialisation
wandb.init(project="PNS Denoising",
           config = {
              "learning_rate": 0.001,
              "epochs": 20,
              "batch_size": 32,
              "kernel_size": 3})

config = wandb.config

# Store in dataloaders
train_dataloader = DataLoader(x_windows, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(x_windows, batch_size=1, shuffle=False)

print("Setting up Noise2Noise model...")

encoder = Noise2NoiseEncoder(num_channels=1)
decoder = Noise2NoiseDecoder(num_channels=1, data_length=len(x_noisy))
model = Noise2NoiseModel(encoder=encoder, decoder=decoder).to(device)

# summary(model, [(1, samples)])

# --- Hyperparameter setup
mse_loss = nn.MSELoss()

def loss_function(x, x_hat):
    return mse_loss(x, x_hat)


optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

# --- Training
print("Start training Noise2Noise model...")
model.train()

# x = torch.from_numpy(x).to(device).float()
# x = torch.unsqueeze(x, 0)
# x = torch.unsqueeze(x, 0) # Expanding to (B, C, L)
# x.requires_grad = True

# Set up toy problem for PyTorch
if torch.cuda.is_available():
    model.cuda()

best_loss = 99999.0

for epoch in range(config.epochs):
    overall_loss = 0

    for batch_idx, (data, target) in enumerate(train_dataloader):
        data = data.unsqueeze(1).to(device).float()
        target = target.unsqueeze(1).to(device).float()
        
        optimizer.zero_grad()
        
        # if batch_idx==1:
        #     plt.plot(data[0,0,:].cpu())
        #     plt.plot(target[0,0,:].cpu())
        #     plt.show()
        
        x_hat = model(data)
        loss = loss_function(x_hat, target)
        
        overall_loss += loss.item()*len(data)
        
        loss.backward()

        # # Uncomment to print out model parameters (check if gradients are there)
        # for param in model.parameters():
        #     print(param.grad)
            
        optimizer.step()

        if loss < best_loss:
            best_model = copy.deepcopy(model)
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*config.batch_size))
    wandb.log({"loss": overall_loss})
        
print("Finished!")

# --- Inference
best_model.eval()

with torch.no_grad():
    best_model.training = False
    x_hats = np.zeros((len(x_windows), samples))

    for idx, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.unsqueeze(1).to(device).float()
        x_hat = best_model(test_data)

        # Convert to numpy and send to cpu
        x_hat = x_hat.cpu().numpy()
        x_hats[idx, :] = x_hat[0]

plt.plot(x_windows[0][0], label="Noisy input")
plt.plot(x_windows[0][1], label="Less noisy input")
plt.plot(x_clean[0:2048], label="Clean sine wave")
# plt.plot(x_medium.detach().cpu(), label="Noisy target")
plt.plot(x_hats.flatten()[0:2048], label="Reconstruction")
plt.legend()
plt.show()