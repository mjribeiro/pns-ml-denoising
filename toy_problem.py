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
fs = 8000
f = 8
samples = 2048
x = np.sin(2 * np.pi * f * np.arange(samples) / fs)

# --- Extra: add noise

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
              "learning_rate": 1e-3,
              "epochs": 5,
              "batch_size": 1,
              "kernel_size": 5,
              "mse_weight": 1})

config = wandb.config

print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=1, latent_dim=1, kernel_size=config.kernel_size, num_layers=2, pool_step=2, device=device)
decoder = Decoder(latent_dim=1, output_dim=1, kernel_size=config.kernel_size, num_layers=2, pool_step=2, device=device)
model = CoordinateVAEModel(Encoder=encoder, Decoder=decoder)

# Hyperparameter setup
mse_loss = nn.MSELoss(reduction='mean')
kld_loss = nn.KLDivLoss(reduction='mean')

def loss_function(x, x_hat, mse_weight=0.25):
    kld_weight = 1 - mse_weight

    mse = mse_loss(x, x_hat)
    kld = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    return (mse_weight * mse) + (kld_weight * -kld)


optimizer = Adam(model.parameters(), lr=config.learning_rate)
wandb.watch(model, log="all")

# --- Training - 1 epoch?
print("Start training VAE...")
model.train()

# Set up toy problem for PyTorch
x = torch.from_numpy(x).float().to(device)
x = torch.unsqueeze(x, 0)
x = torch.unsqueeze(x, 0)
x.requires_grad = True

for epoch in range(config.epochs):
    overall_loss = 0
    model.epoch = epoch

    optimizer.zero_grad()

    x_hat = model(x)
    loss = loss_function(x, x_hat, mse_weight=config.mse_weight)
    
    overall_loss += loss.item()
    
    loss.backward()
    for i in range(2):
        print("-----------------------------------------------------")
        print(f"Encoder weights in layer {i}: {model.Encoder.layers[i].weight}")
        print(f"Decoder weights in layer {i}: {model.Decoder.layers[i].weight}")
        print(f"Encoder grad for weights in layer {i}: {model.Encoder.layers[i].weight.grad}")
        print(f"Decoder grad for weights in layer {i}: {model.Decoder.layers[i].weight.grad}")
    optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tLoss: ", loss.item())
    wandb.log({"loss": loss.item()})
        
print("Finished!")

# --- Inference
model.eval()

with torch.no_grad():
    model.training = False
    x_hat = model(x)

# x = x.view(16, 1, input_dim)
plt.plot(x.detach().numpy()[0][0])
plt.plot(x_hat[0][0])
plt.show()