import copy
import matplotlib.pyplot as plt
import numpy as np
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
f = 500
samples = 256
x = np.sin(2 * np.pi * f * np.arange(samples) / fs)

# --- Extra: add noise
# noise_levels = np.arange(0.1, 0.6, 0.1)
# input_data = np.zeros((len(noise_levels), samples))

# for i in range(len(noise_levels)):
#     input_data[i] = x + np.random.normal(0, noise_levels[i], samples)

# input_data = np.asarray(input_data)
# time_samples = np.where(input_data==input_data.min())

# --- Plot resulting sine waveform
# plt.plot(np.arange(samples), x)
# plt.xlabel('Samples')
# plt.ylabel('Voltage (V)')
# plt.show()

# --- Set up model
# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Weights&Biases initialisation
wandb.init(project="PNS Denoising",
           config = {
              "learning_rate": 0.01,
              "epochs": 1000,
              "batch_size": 1,
              "kernel_size": 3,
              "mse_weight": 1})

config = wandb.config

print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=1, latent_dim=1, kernel_size=config.kernel_size, num_layers=3, pool_step=2, device=device)
decoder = Decoder(latent_dim=1, output_dim=1, kernel_size=config.kernel_size, num_layers=3, pool_step=2, device=device)
model = CoordinateVAEModel(Encoder=encoder, Decoder=decoder)

# --- Hyperparameter setup
mse_loss = nn.MSELoss()
kld_loss = nn.KLDivLoss()

def loss_function(x, x_hat, mse_weight=0.25):
    kld_weight = 1 - mse_weight

    mse = mse_loss(x, x_hat)
    kld = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    return (mse_weight * mse) + (kld_weight * -kld)


optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
wandb.watch(model, log="all")

# --- Training - 1 epoch?
print("Start training VAE...")
model.train()

# Set up toy problem for PyTorch
if torch.cuda.is_available():
    model.cuda()

# x = torch.from_numpy(input_data).float().to(device)
x = torch.from_numpy(x).float().to(device)
x = torch.unsqueeze(x, 0)
x = torch.unsqueeze(x, 0) # Expanding to (B, C, L) format (B=batch size, C=channels, L=length)
x.requires_grad = True

best_loss = 99999.0

for epoch in range(config.epochs):
    overall_loss = 0
    model.epoch = epoch + 1

    optimizer.zero_grad()

    x_hat = model(x)
    loss = loss_function(x, x_hat, mse_weight=config.mse_weight)
    
    overall_loss += loss.item()*len(x)
    
    loss.backward()

    # print("-----------------------------------------------------")
    # for i in range(2):
    #     print(f"Encoder weights in layer {i}: {model.Encoder.layers[i].weight}")
    #     print(f"Decoder weights in layer {i}: {model.Decoder.layers[i].weight}")
    #     print(f"Encoder grad for weights in layer {i}: {model.Encoder.layers[i].weight.grad}")
    #     print(f"Decoder grad for weights in layer {i}: {model.Decoder.layers[i].weight.grad}")
    #     print("\n ----- \n")
        
    optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tLoss: ", loss.item())
    wandb.log({"Epoch loss": loss.item()})

    if loss < best_loss:
        best_model = copy.deepcopy(model)
        
print("Finished!")

# --- Inference
best_model.eval()

with torch.no_grad():
    best_model.training = False
    x_hat = best_model(x)

# x = x.view(16, 1, input_dim)
plt.plot(x.cpu().detach().numpy()[0][0])
plt.plot(x_hat.cpu()[0][0])
plt.show()