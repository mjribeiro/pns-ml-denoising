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

# Local imports
from models.vae_model import *

# --- Create sine wave
fs = 30000
f = 10
total_samples = 10000
num_repeats = 1

x = np.zeros((num_repeats, total_samples))
x_delayed = np.zeros((num_repeats, total_samples))

for i in range(num_repeats):
    x[i, :] = np.sin(2 * np.pi * f * np.arange(total_samples) / fs)
    x_delayed[i, :] = np.roll(x[i, :], 1000)

x_flat = x.flatten()
x_delayed_flat = x_delayed.flatten()

# # Plot two-channel toy problem
# plt.plot(x_flat)
# plt.plot(x_delayed_flat)
# plt.show()

# Take smaller windows from main sine signal
samples = 128
x_windows = []
for i in range(0, len(x_flat) - samples, samples):
    x_windows.append(np.asarray([x_flat[i:i+samples], x_delayed_flat[i:i+samples]]))
x_windows = np.asarray(x_windows)

# # Check shape of data going in
# print(x_windows.shape)
# plt.plot(x_windows[10, 0, :])
# plt.plot(x_windows[10, 1, :])
# plt.show()

# Store in dataloaders
train_dataloader = DataLoader(x_windows, batch_size=8, shuffle=True)
test_dataloader = DataLoader(x_windows, batch_size=1, shuffle=False)

# # --- Add overlap
# overlap_amount = 0.5
# overlapping_x = []
# step_size = int(samples * overlap_amount)
# overlapping_x = []

# for step in range(0, len(x_flat) - step_size, step_size):
#     overlapping_x.append(x_flat[step:step+int(step_size/overlap_amount)])

# # --- Plot resulting sine waveform
# plt.plot(np.arange(samples * num_windows), x.flatten())
# plt.xlabel('Samples')
# plt.ylabel('Voltage (V)')
# plt.show()

# --- Set up model
# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Weights&Biases initialisation
wandb.init(project="PNS Denoising",
           config = {
              "learning_rate": 0.0005,
              "epochs": 5000,
              "batch_size": 1,
              "kernel_size": 3,
              "mse_weight": 1})

config = wandb.config

print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=1, latent_dim=1, kernel_size=config.kernel_size, num_layers=2, pool_step=2, device=device)
decoder = Decoder(latent_dim=1, output_dim=1, kernel_size=config.kernel_size, num_layers=2, pool_step=2, device=device)
model = CoordinateVAEModel(Encoder=encoder, Decoder=decoder)

# --- Hyperparameter setup
mse_loss = nn.MSELoss()
kld_loss = nn.KLDivLoss()

def loss_function(x, x_hat, mse_weight=0.25):
    kld_weight = 1 - mse_weight

    mse = mse_loss(x, x_hat)
    kld = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    return (mse_weight * mse) + (kld_weight * -kld)


optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
wandb.watch(model, log="all")

# --- Training - 1 epoch?
print("Start training VAE...")
model.train()

# Set up toy problem for PyTorch
if torch.cuda.is_available():
    model.cuda()

best_loss = 99999.0

for epoch in range(config.epochs):
    overall_loss = 0
    model.epoch = epoch + 1

    for batch_idx, data in enumerate(train_dataloader):
        data = data.to(device).float()
        # data = data.unsqueeze(1) # Expanding to (B, C, L)
        
        optimizer.zero_grad()

        x_hat = model(data)
        loss = loss_function(data, x_hat, mse_weight=config.mse_weight)
        
        overall_loss += loss.item()*len(data)
        
        loss.backward()

        # # Uncomment to print out model parameters (check if gradients are there)
        # for param in model.parameters():
        #     print(param.grad)
            
        optimizer.step()

        if loss < best_loss:
            best_model = copy.deepcopy(model)
            
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*8))
    wandb.log({"loss": overall_loss / (batch_idx*8)})
        
print("Finished!")

# --- Inference
best_model.eval()

with torch.no_grad():
    best_model.training = False
    x_hats = np.zeros((len(x_windows), 2, samples))
    # onehots = np.zeros((len(x_windows), int(samples/4)))
    for idx, test_data in enumerate(test_dataloader):
        test_data = test_data.to(device).float()
        x_hat, onehot = best_model(test_data)

        # Convert to numpy and send to cpu
        x_hat = x_hat.cpu().numpy()
        # onehot = onehot.cpu().numpy()

        # Store in x_hats
        x_hats[idx, :, :] = x_hat[0]
        # onehots[idx, :] = onehot

plt.plot(x.flatten(), label="original sine")
plt.plot(x_delayed_flat, label="delayed sine")
plt.plot(x_hats[:, 0, :].flatten(), label="reconstructed original sine")
plt.plot(x_hats[:, 1, :].flatten(), label="reconstructed delayed sine")
plt.legend()
plt.show()