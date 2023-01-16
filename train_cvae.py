
import copy
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader
# from torchsummary import summary

# Temp
import matplotlib.pyplot as plt

# Local imports
from models.cvae_model import *
from datasets.vagus_dataset import VagusDataset, VagusDatasetN2N

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
# train_dataset = VagusDataset(train=True)
# test_dataset  = VagusDataset(train=False)

# train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
# test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

train_dataset = VagusDatasetN2N(stage="train")
test_dataset  = VagusDatasetN2N(stage="test")

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

# sample = train_dataset.__getitem__(0)

# # Plot single channel
# plt.plot(sample[0, 0, :])
# plt.show()

# Define model
print("Setting up coordinate VAE model...")

# Get sample from dataset for figuring out number of channels etc automatically :)
sample = next(iter(train_dataloader))[0]

encoder = Encoder(input_dim=sample.shape[1],
                  kernel_size=config.kernel_size, 
                  num_layers=7, 
                  pool_step=2, 
                  batch_size=config.batch_size, 
                  device=device)

# Get latent data size for coordinate encoder - depends on number of layers and pool size
# of main encoder, since differing amounts of maxpooling may be used
with torch.no_grad():
    x = torch.randn_like(sample).float()
    test_encoder_output, _   = encoder(x)
    coord_encoder_output_dim = test_encoder_output.shape[-1]

coordinate_encoder = CoordinateEncoder(input_dim=sample.shape[1],
                                       output_dim=coord_encoder_output_dim,
                                       all_samples=10,
                                       subset_samples=5,
                                       kernel_size=3,
                                       num_layers=4,
                                       pool_step=4,
                                       device=device)
coordinate_encoder.to(device)

# Get size of coordinate encoder output
with torch.no_grad():
    _, sampled_data   = coordinate_encoder(test_encoder_output)
    decoder_input_dim = sampled_data.shape[1]

decoder = Decoder(latent_dim=decoder_input_dim+test_encoder_output.shape[1], # TODO: Include this in model instead 
                  output_dim=sample.shape[1], 
                  kernel_size=config.kernel_size, 
                  num_layers=7, 
                  pool_step=2, 
                  device=device)

model = CoordinateVAEModel(encoder=encoder, 
                           decoder=decoder,
                           coordinate_encoder=coordinate_encoder,
                           device=device)

# Hyperparameter setup
mse_loss = nn.MSELoss()
kld_loss = nn.KLDivLoss()

def loss_function(x, x_hat, kld_weight):

    MSE = mse_loss(x, x_hat)
    KLD = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    MSE = (1 - kld_weight) * MSE
    KLD = kld_weight * KLD
    
    return MSE + KLD, KLD


optimizer = Adam(model.parameters(), 
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

for epoch in range(config.epochs):
    overall_loss = 0
    # TODO: Check if epoch should be zero-indexed or not - doesn't seem to make a difference?
    model.epoch = epoch

    for batch_idx, (_, x) in enumerate(train_dataloader):
        x = x.to(device).float()

        optimizer.zero_grad()

        x_hat = model(x)
        loss, kld = loss_function(x, x_hat, kld_weight=kld_weight)

        if loss < best_loss:
            best_model = copy.deepcopy(model)
        
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
model.training = False

# TODO: TEST ON SAME DATA AS NOISE2NOISE

# Inference
model.eval()

with torch.no_grad():
    x_hats = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
    xs = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
    bp = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
    
    start_idx = 0
    end_idx = start_idx+config.batch_size

    for batch_idx, (_, x) in enumerate(test_dataloader):
        x = x.to(device).float()
        model.training = False
        
        x_hat = model(x)
        x_hat = x_hat.cpu().numpy()

        x_hats[start_idx:end_idx, :, :] = x_hat
        xs[start_idx:end_idx, :, :] = x.cpu().numpy()
        bp[start_idx:end_idx, :, :] = test_dataset.load_bp_data(start_idx, end_idx)

        start_idx += config.batch_size
        end_idx += config.batch_size

# x = x.view(16, 1, input_dim)
fig, ax = plt.subplots()
# ax = plt.gca()

start_plot_sample = 1000
end_plot_sample = 1004

time = np.arange(0, len(xs[start_plot_sample:end_plot_sample, 0, :].flatten())/100e3, 1/100e3)

ax.plot(time, xs[start_plot_sample:end_plot_sample, 0, :].flatten(), label="Noisy input")
ax.plot(time, x_hats[start_plot_sample:end_plot_sample, 0, :].flatten(), label="Reconstructed")
# ax.plot(time, bp[:, 0, :].flatten())
# ax.xlabel("Time (s)")
# ax.ylabel("Amplitude (AU, normalised)")

# plt.rcParams.update({'font.size': 22})
# plt.show()

wandb.log({"plot": fig})

wandb.finish()