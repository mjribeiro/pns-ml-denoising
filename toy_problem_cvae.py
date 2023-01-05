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
from models.cvae_model import *

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
              "epochs": 500,
              "batch_size": 32,
              "kernel_size": 3})

config = wandb.config

# def train():
# Store in dataloaders
train_dataloader = DataLoader(x_windows, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(x_windows, batch_size=1, shuffle=False)

print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=2, 
                latent_dim=100, 
                kernel_size=config.kernel_size, 
                num_layers=3, 
                pool_step=4, 
                batch_size=config.batch_size, 
                device=device)
decoder = Decoder(latent_dim=100, 
                output_dim=2, 
                kernel_size=config.kernel_size, 
                num_layers=3, 
                pool_step=4, 
                device=device)
model = CoordinateVAEModel(encoder=encoder, 
                           decoder=decoder)

# # -- View model
# summary(model.to(device), [(2, samples)], 1)

# --- Hyperparameter setup
mse_loss = nn.MSELoss(reduction='mean')
kld_loss = nn.KLDivLoss()

def loss_function(x, x_hat, kld_weight):

    MSE = mse_loss(x, x_hat)
    KLD = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    # log_ratio = torch.log(q_y * categorical_dim + 1e-20)
    # KLD = torch.sum(q_y * log_ratio, dim=-1).mean()
    # KLD = torch.sum(q_y * log_ratio, dim=-1).mul_(-0.5)
    # print("KLD was: ", KLD)

    MSE = (1 - kld_weight) * MSE
    KLD = kld_weight * KLD
    
    # return (mse_weight * mse) + (kld_weight * -kld)
    return MSE + KLD, KLD


optimizer = AdamW(model.parameters(), 
                lr=config.learning_rate,
                weight_decay=1e-5)
wandb.watch(model, log="all")

# --- Training
print("Start training VAE...")
model.train()

# Set up toy problem for PyTorch
if torch.cuda.is_available():
    model.cuda()

best_loss = 99999.0
kld_weight = 0.7
kld_rate = 0 # TODO: Change this to rate of increase as per:
                    # https://arxiv.org/pdf/1511.06349.pdf
kld_tracked = []
kld_w_tracked = []

for epoch in range(config.epochs):
    overall_loss = 0
    model.epoch = epoch

    for batch_idx, data in enumerate(train_dataloader):
        data = data.to(device).float()
        # data = data.unsqueeze(1) # Expanding to (B, C, L)
        
        optimizer.zero_grad()

        x_hat = model(data)
        loss, kld = loss_function(data, x_hat, kld_weight=kld_weight)

        if loss < best_loss:
            best_model = copy.deepcopy(model)
            
        kld_tracked.append(kld.detach().cpu())
        kld_w_tracked.append(kld_weight)

        overall_loss += loss.item()*len(data)
        
        loss.backward()
            
        optimizer.step()

        if kld_weight < 1:
            kld_weight += kld_rate

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
        x_hat = best_model(test_data)

        # Convert to numpy and send to cpu
        x_hat = x_hat.cpu().numpy()
        # onehot = onehot.cpu().numpy()

        # Store in x_hats
        x_hats[idx, :, :] = x_hat[0]
        # onehots[idx, :] = onehot


# Reconstruction
plt.plot(x.flatten(), label="original sine")
plt.plot(x_delayed_flat, label="delayed sine")
plt.plot(x_hats[:, 0, :].flatten(), label="reconstructed original sine")
plt.plot(x_hats[:, 1, :].flatten(), label="reconstructed delayed sine")
plt.legend()
plt.show()

# KLD analysis
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.set_ylabel("KLD", color='r')
ax2.set_ylabel("KLD weight", color='b')

ax2.set_ylim((min(kld_w_tracked) - 0.2, max(kld_w_tracked) + 0.2))

ax.plot(kld_tracked, "r-")
ax2.plot(kld_w_tracked, "b-")
plt.legend()
plt.show()

# # Hyperparameter opt
# sweep_configuration = {
#     'method': 'bayes',
#     'metric': {
#         'goal': 'minimize', 
#         'name': 'loss'
#         },
#     'parameters': {
#         'batch_size': {'values': [8, 16, 32, 64]},
#         'epochs': {'values': [5, 10, 15, 20, 25, 30]},
#         'lr': {'max': 0.1, 'min': 0.000001},
#         'mse_weight': {'max': 0.9, 'min': 0.1}
#     }
# }

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="PNS Denoising")

# wandb.agent(sweep_id, train)