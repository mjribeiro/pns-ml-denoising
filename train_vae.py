import copy
import gc
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import AdamW
from torch.utils.data import DataLoader

# Temp
import matplotlib.pyplot as plt

# Local imports
from models.vae import *
from datasets.vagus_dataset import VagusDataset, VagusDatasetN2N
from utils.analysis import *



torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Address GPU memory issues (source: https://stackoverflow.com/a/66921450)
gc.collect()
torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)

# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Weights&Biases initialisation
wandb.init(project="PNS Denoising",
        config = {
            "learning_rate": 5e-3,
            "epochs": 2000,
            "batch_size": 2048,
            "kernel_size": 5,
            "change_weights": False,
            "latent_dim": 120,
            "early_stop": False})

config = wandb.config

# Load vagus dataset, using filtered data from Noise2Noise as inputs (see training loop later)
# train_dataset = VagusDataset(train=True)
# test_dataset  = VagusDataset(train=False)
train_dataset = VagusDatasetN2N(stage="train")
val_dataset   = VagusDatasetN2N(stage="val")
test_dataset  = VagusDatasetN2N(stage="test")

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
val_dataloader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_dataloader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

# sample = train_dataset.__getitem__(0)

# # Plot single channel
# plt.plot(sample[0, 0, :])
# plt.show()

# Define model
print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=9, 
                  latent_dim=config.latent_dim, 
                  kernel_size=config.kernel_size, 
                  num_layers=4, 
                  pool_step=2, 
                  batch_size=config.batch_size, 
                  device=device)
decoder = Decoder(latent_dim=config.latent_dim, 
                  output_dim=9, 
                  kernel_size=config.kernel_size, 
                  num_layers=4, 
                  pool_step=2, 
                  device=device)
model = CoordinateVAEModel(Encoder=encoder, 
                           Decoder=decoder)

# Hyperparameter setup
mse_loss = nn.MSELoss(reduction='mean')
kld_loss = nn.KLDivLoss(reduction='batchmean')

def loss_function(x, x_hat, bp, device, epoch, idx, change_weights=False):

    MSE = mse_loss(x, x_hat)
    KLD = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    # Adapt loss based on current epoch
    if change_weights:
        loss_step = config.epochs - (config.epochs / 4)
        loss_weight = np.linspace(0.01, 0.5, int(loss_step))

        if epoch < (config.epochs / 4):
            loss = (0.5 * MSE) + (0.5 * KLD)
        else:
            # Get moving RMS plots
            bp_envelope, x_hat_moving_rms = get_rms_envelope(x_hat=x_hat, bp=bp, window_len=int(100e3), device=device)

            MSE_envelope = mse_loss(bp_envelope, x_hat_moving_rms)

            MSE_envelope_weight = loss_weight[idx]
            MSE_reconstr_weight = 0.5 * (1 - MSE_envelope_weight)
            KLD_weight = 0.5 * (1 - MSE_envelope_weight)

            loss =  (MSE_envelope_weight * MSE_envelope) + (MSE_reconstr_weight * MSE) + (KLD_weight * KLD)
    else:
        # Get moving RMS plots
        bp_envelope, x_hat_moving_rms = get_rms_envelope(x_hat=x_hat, bp=bp, window_len=int(100e3), device=device)
        MSE_envelope = mse_loss(bp_envelope, x_hat_moving_rms)

        MSE_envelope_weight = 0.5
        KLD_weight          = 0.375
        MSE_reconstr_weight = 1 - MSE_envelope_weight - KLD_weight

        loss =  (MSE_envelope_weight * MSE_envelope) + (MSE_reconstr_weight * MSE) + (KLD_weight * KLD)
    
    return loss, KLD


optimizer = AdamW(model.parameters(), 
                  lr=config.learning_rate,
                  weight_decay=0.01)
wandb.watch(model, log="all")

# Training
print("Start training VAE...")

if torch.cuda.is_available():
    model.cuda()

# Get parameter count
print(sum(p.numel() for p in model.parameters()))

training_start_time = time.time()

best_loss = 99999.0
best_loss_epoch = 0
# kld_weight = 0.5
# kld_rate = 0 # TODO: Could change this to rate of increase as per:
#                     # https://arxiv.org/pdf/1511.06349.pdf

idx = 0

for epoch in range(config.epochs):
    model.train()
    overall_loss = 0
    model.epoch = epoch

    # ----- TRAIN -----
    # (_, x) since N2N data has raw signal and filtered signal, so for VAE only using filtered
    # signal as the input
    for batch_idx, (_, x, bp) in enumerate(train_dataloader):
        x = x.to(device).float()

        optimizer.zero_grad()

        x_hat = model(x)
        loss, _ = loss_function(x, x_hat, bp, device=device, epoch=epoch, idx=idx, change_weights=config.change_weights)
        
        overall_loss += loss.item() * x.size(0)
        
        loss.backward()
        optimizer.step()
    
    average_loss = overall_loss / len(train_dataset)
    
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", average_loss)
    wandb.log({"loss": average_loss})

    # ----- VALIDATE -----
    model.eval()

    with torch.no_grad():
        overall_val_loss = 0
        for batch_idx, (_, x, bp) in enumerate(val_dataloader):
            x = x.to(device).float()

            x_hat = model(x)
            loss, _ = loss_function(x, x_hat, bp, device=device, epoch=epoch, idx=idx, change_weights=config.change_weights)

            overall_val_loss += loss.item() * x.size(0)

        average_val_loss = overall_val_loss / len(val_dataset)
        print("\t\t\t\tAverage Validation Loss: ", average_val_loss)
        wandb.log({"val_loss": average_val_loss})

    # Index that updates with each epoch past a certain point
    # Needed for updating weights for loss elements
    if epoch >= (config.epochs / 4): 
        idx += 1

    if config.early_stop:
        if loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = average_loss
            best_loss_epoch = epoch

        if epoch > best_loss_epoch + config.epochs // 5:
            break
        
print("Finished!")
print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))

# TODO: Folder needs to be created/checked if exists before using torch.save()
if config.early_stop:
    model = best_model
    
PATH = './saved/coordinate_vae.pth'
torch.save(model.state_dict(), PATH)

# model = best_model
model.training = False

# Inference
model.eval()

with torch.no_grad():
    x_hats = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
    xs = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
    bp_arr = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
    
    start_idx = 0
    end_idx = start_idx+config.batch_size

    # Test on noisy data rather than BP filtered
    for batch_idx, (_, x, bp) in enumerate(test_dataloader):
        x = x.to(device).float()
        model.training = False
        
        x_hat = model(x)
        x_hat = x_hat.cpu().numpy()

        x_hats[start_idx:end_idx, :, :] = x_hat
        xs[start_idx:end_idx, :, :] = x.cpu().numpy()
        bp_arr[start_idx:end_idx, :, :] = test_dataset.load_bp_data(start_idx, end_idx)

        start_idx += config.batch_size
        end_idx += config.batch_size

# x = x.view(16, 1, input_dim)
ax = plt.gca()

time = np.arange(0, len(xs[:, 0, :].flatten())/100e3, 1/100e3)
plt.plot(time, xs[:, 0, :].flatten())
plt.plot(time, x_hats[:, 0, :].flatten())
plt.plot(time, bp_arr[:, 0, :].flatten())
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (AU, normalised)")

plt.rcParams.update({'font.size': 22})
plt.show()

for i in range(9):
    np.save(f"./results/cvae_noisy_input_ch{i+1}.npy", xs[:, i, :].flatten())
    np.save(f"./results/cvae_reconstr_rms_ch{i+1}.npy", x_hats[:, i, :].flatten())

wandb.finish()