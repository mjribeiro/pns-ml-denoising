import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import AdamW, SGD
from torch.utils.data import DataLoader
from torchsummary import summary

# Local imports
from models.vae_model import *

# --- Create sine wave
fs = 30000
f = 7
total_samples = 10000
num_repeats = 1

x = np.zeros((num_repeats, total_samples))
x_delayed = np.zeros((num_repeats, total_samples))

add_noise = True

for i in range(num_repeats):
    x[i, :] = np.sin(2 * np.pi * f * np.arange(total_samples) / fs)
    x_delayed[i, :] = np.roll(x[i, :], 3000)

    if add_noise:
        x[i, :] = x[i, :] + np.random.normal(0, 0.08, total_samples)
        x_delayed[i, :] = x_delayed[i, :] + np.random.normal(0, 0.08, total_samples)

x_flat = x.flatten()
x_delayed_flat = x_delayed.flatten()

# # Plot two-channel toy problem
# plt.plot(x_flat)
# plt.plot(x_delayed_flat)
# plt.show()

samples = 256

# # --- Add overlap
# overlap_amount = 0.5
# overlapping_x = []
# step_size = int(samples * overlap_amount)
# overlapping_x = []

# for step in range(0, len(x_flat) - step_size, step_size):
#     overlapping_x.append(x_flat[step:step+int(step_size/overlap_amount)])

# Take smaller windows from main sine signal
x_windows = []
for i in range(0, len(x_flat) - samples, samples):
    x_windows.append(np.asarray([x_flat[i:i+samples], x_delayed_flat[i:i+samples]]))
x_windows = np.asarray(x_windows)

# # Check shape of data going in
# print(x_windows.shape)
# plt.plot(x_windows[10, 0, :])
# plt.plot(x_windows[10, 1, :])
# plt.show()

# --- Set up model
# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Weights&Biases initialisation
wandb.init(project="PNS Denoising",
           config = {
              "learning_rate": 0.001,
              "epochs": 1000,
              "batch_size": 32,
              "kernel_size": 3})

config = wandb.config

# Store in dataloaders
train_dataloader = DataLoader(x_windows, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(x_windows, batch_size=1, shuffle=False)

print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=samples, 
                  latent_dim=100, 
                  kernel_size=config.kernel_size, 
                  num_layers=3, 
                  pool_step=4, 
                  batch_size=config.batch_size, 
                  device=device)
decoder = Decoder(latent_dim=100, 
                  output_dim=samples, 
                  kernel_size=config.kernel_size, 
                  num_layers=3, 
                  pool_step=4, 
                  device=device)
model = CoordinateVAEModel(Encoder=encoder, 
                           Decoder=decoder)

# # -- View model
# summary(model.to(device), [(2, samples)], 1)

# --- Hyperparameter setup
mse_loss = nn.MSELoss()
# kld_loss = nn.KLDivLoss()

def loss_function(x, x_hat, q_y, categorical_dim, kld_weight):

    MSE = mse_loss(x, x_hat)
    # kld = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    log_ratio = torch.log(q_y * categorical_dim + 1e-20)
    KLD = torch.sum(q_y * log_ratio, dim=-1).mean()
    # print("KLD was: ", KLD)

    MSE = (1 - kld_weight) * MSE
    KLD = kld_weight * KLD
    
    # return (mse_weight * mse) + (kld_weight * -kld)
    return MSE + KLD, KLD


optimizer = AdamW(model.parameters(), 
                  lr=config.learning_rate)
wandb.watch(model, log="all")

# --- Training - 1 epoch?
print("Start training VAE...")
model.train()

# Set up toy problem for PyTorch
if torch.cuda.is_available():
    model.cuda()

best_loss = 99999.0
kld_weight = 0.001
kld_rate = 2 # TODO: Change this to rate of increase as per:
                      # https://arxiv.org/pdf/1511.06349.pdf
kld_tracked = []

for epoch in range(config.epochs):
    overall_loss = 0
    # model.epoch = epoch + 1

    for batch_idx, data in enumerate(train_dataloader):
        data = data.to(device).float()
        # data = data.unsqueeze(1) # Expanding to (B, C, L)
        
        optimizer.zero_grad()

        x_hat, q_y, categorical_dim = model(data)
        loss, kld = loss_function(data, x_hat, q_y, categorical_dim, kld_weight=kld_weight)
        kld_tracked.append(kld.detach().cpu())

        overall_loss += loss.item()*len(data)
        
        loss.backward()
            
        optimizer.step()

        if kld_weight < 1:
            kld_weight *= kld_rate

        if kld_weight > 1:
            kld_weight = 1

        # if (epoch % 99 == 0) and (batch_idx == 0):
        #     # # Uncomment to print out model parameters (check if gradients are there)
        #     # for param in model.parameters():
        #     #     print(param.grad)

        #     x_hat = x_hat.detach().cpu().numpy()
        #     plt.plot(x_hat[0, 0, :])
        #     plt.plot(x_hat[0, 1, :])
        #     plt.show(block=False)
        #     plt.pause(3)
        #     plt.close()

        if loss < best_loss:
            best_model = copy.deepcopy(model)
            
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*config.batch_size))
    wandb.log({"loss": overall_loss / (batch_idx*config.batch_size)})
        
print("Finished!")

# --- Inference
best_model.eval()

with torch.no_grad():
    best_model.training = False
    x_hats = np.zeros((len(x_windows), 2, samples))
    # onehots = np.zeros((len(x_windows), int(samples/4)))
    for idx, test_data in enumerate(test_dataloader):
        test_data = test_data.to(device).float()
        x_hat, q_y, categorical_dim = best_model(test_data)

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

plt.plot(kld_tracked)
plt.show()