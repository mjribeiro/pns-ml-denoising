import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import Adam
from torch.utils.data import DataLoader

# Temp
import matplotlib.pyplot as plt

# Local imports
from models.vae_model import *
from datasets.vagus_dataset import VagusDataset

# Weights&Biases initialisation
wandb.init(project="PNS Denoising",
           config = {
               "learning_rate": 1e-3,
               "epochs": 10,
               "batch_size": 16})

config = wandb.config

# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load vagus dataset
train_dataset = VagusDataset(train=True)
test_dataset  = VagusDataset(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

# sample = train_dataset.__getitem__(0)[0]

# plt.plot(sample)
# plt.show()

# Define model
input_dim = 2048 # TODO: don't hardcode this, find out from input
hidden_dim_encoder = 256
hidden_dim_decoder = 256
kernel_size = 1

print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=1, latent_dim=1, kernel_size=kernel_size, device=device)
decoder = Decoder(latent_dim=1, output_dim=1, kernel_size=kernel_size, device=device)
model = CoordinateVAEModel(Encoder=encoder, Decoder=decoder, device=device)

# Hyperparameter setup
mse_loss = nn.MSELoss(reduction='sum')
kld_loss = nn.KLDivLoss(reduction='sum')

def loss_function(x, x_hat, mse_weight=0.5, kld_weight=0.5):
    mse = mse_loss(x, x_hat)
    kld = kld_loss(F.log_softmax(x, -1), F.softmax(x_hat, -1))

    return (mse_weight * mse) + (kld_weight * -kld)


optimizer = Adam(model.parameters(), lr=config.learning_rate)
wandb.watch(model)

# Training
print("Start training VAE...")
model.train()

for epoch in range(config.epochs):
    overall_loss = 0
    model.epoch = epoch

    for batch_idx, x in enumerate(train_dataloader):
        x = x.to(device).float()

        optimizer.zero_grad()

        x_hat = model(x)
        loss = loss_function(x, x_hat)
        
        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*config.batch_size))
    wandb.log({"loss": overall_loss / (batch_idx*config.batch_size)})
        
print("Finished!")

# TODO: Folder needs to be created/checked if exists before using torch.save()
PATH = './saved/coordinate_vae.pth'
torch.save(model.state_dict(), PATH)

wandb.finish()
