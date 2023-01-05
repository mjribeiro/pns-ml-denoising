
import copy
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchsummary import summary

# Temp
import matplotlib.pyplot as plt

# Local imports
from models.cvae_model import *
from datasets.vagus_dataset import VagusDataset

# Address GPU memory issues (source: https://stackoverflow.com/a/66921450)
gc.collect()
torch.cuda.empty_cache()

# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Weights&Biases initialisation
wandb.init(project="PNS Denoising",
        config = {
            "learning_rate": 0.001,
            "epochs": 2,
            "batch_size": 32,
            "kernel_size": 3})

config = wandb.config

# Load vagus dataset
train_dataset = VagusDataset(train=True)
test_dataset  = VagusDataset(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

# sample = train_dataset.__getitem__(0)

# # Plot single channel
# plt.plot(sample[0, 0, :])
# plt.show()

# Define model
print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=9, 
                latent_dim=100, 
                kernel_size=config.kernel_size, 
                num_layers=4, 
                pool_step=4, 
                batch_size=config.batch_size, 
                device=device)
decoder = Decoder(latent_dim=100, 
                output_dim=9, 
                kernel_size=config.kernel_size, 
                num_layers=4, 
                pool_step=4, 
                device=device)
model = CoordinateVAEModel(encoder=encoder, 
                           decoder=decoder)

# Hyperparameter setup
mse_loss = nn.MSELoss()
kld_loss = nn.KLDivLoss()

def loss_function(x, x_hat, kld_weight):

    MSE = mse_loss(x, x_hat)
    KLD = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    MSE = (1 - kld_weight) * MSE
    KLD = kld_weight * KLD
    
    return MSE + KLD, KLD


optimizer = AdamW(model.parameters(), 
                lr=config.learning_rate,
                weight_decay=1e-5)
wandb.watch(model, log="all")

# summary(model, (9, 1024), device='cpu')

# Training
print("Start training VAE...")

if torch.cuda.is_available():
    model.cuda()

# ----- TRAINING -----
model.train()

best_loss = 99999.0
kld_weight = 0.5
kld_rate = 0 # TODO: Change this to rate of increase as per:
                    # https://arxiv.org/pdf/1511.06349.pdf
# kld_tracked = []
# kld_w_tracked = []

for epoch in range(config.epochs):
    overall_loss = 0
    # TODO: Check if epoch should be zero-indexed or not - doesn't seem to make a difference?
    model.epoch = epoch

    for batch_idx, x in enumerate(train_dataloader):
        x = x.to(device).float()

        optimizer.zero_grad()

        x_hat = model(x)
        loss, kld = loss_function(x, x_hat, kld_weight=kld_weight)

        if loss < best_loss:
            best_model = copy.deepcopy(model)

        # kld_tracked.append(kld.detach().cpu())
        # kld_w_tracked.append(kld_weight)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*config.batch_size))
    wandb.log({"loss": overall_loss / (batch_idx*config.batch_size)})
        
print("Finished!")

Path('./saved/').mkdir(parents=True, exist_ok=True)
torch.save(best_model.state_dict(), './saved/coordinate_vae.pth')

# ----- INFERENCE -----
# model.load_state_dict(torch.load('./saved/coordinate_vae.pth'))
model = best_model

# Inference
model.eval()

with torch.no_grad():
    x_hats = np.zeros((len(test_dataloader), 9, 1024))
    xs = np.zeros((len(test_dataloader), 9, 1024))
    for batch_idx, x in enumerate(test_dataloader):
        x = x.to(device).float()
        model.training = False
        
        x_hat = model(x)
        x_hat = x_hat.cpu().numpy()

        x_hats[batch_idx, :, :] = x_hat[0]
        xs[batch_idx, :, :] = x[0].cpu().numpy()

# x = x.view(16, 1, input_dim)
ax = plt.gca()

time = np.arange(0, len(xs[:, 0, :].flatten())/100e3, 1/100e3)
plt.plot(time, xs[:, 0, :].flatten())
plt.plot(time, x_hats[:, 0, :].flatten())
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (AU, normalised)")

plt.rcParams.update({'font.size': 22})
plt.show()

wandb.finish()