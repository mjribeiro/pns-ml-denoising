import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import AdamW
from torch.utils.data import DataLoader

# Temp
import matplotlib.pyplot as plt

# Local imports
from models.vae_model import *
from datasets.vagus_dataset import VagusDataset

# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Weights&Biases initialisation
wandb.init(project="PNS Denoising",
        config = {
            "learning_rate": 1e-3,
            "epochs": 10,
            "batch_size": 16,
            "kernel_size": 3,
            "latent_dim": 100,
            "kld_weight": 0.5,
            "num_layers": 3,
            "pool_step": 2})

config = wandb.config

# Load vagus dataset
train_dataset = VagusDataset(train=True)
test_dataset  = VagusDataset(train=False)

def train():
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # sample = train_dataset.__getitem__(0)[0]

    # plt.plot(sample)
    # plt.show()

    # Define model
    input_dim = 2048 # TODO: don't hardcode this, find out from input
    kernel_size = 1

    print("Setting up coordinate VAE model...")
    encoder = Encoder(input_dim=9, 
                latent_dim=config.latent_dim, 
                kernel_size=config.kernel_size, 
                num_layers=config.num_layers, 
                pool_step=config.pool_step, 
                batch_size=config.batch_size, 
                device=device)
    decoder = Decoder(latent_dim=config.latent_dim, 
                    output_dim=9, 
                    kernel_size=config.kernel_size, 
                    num_layers=config.num_layers, 
                    pool_step=config.pool_step, 
                    device=device)
    model = CoordinateVAEModel(Encoder=encoder, 
                            Decoder=decoder)

    # Hyperparameter setup
    mse_loss = nn.MSELoss(reduction='mean')
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
            loss, kld = loss_function(x, x_hat, kld_weight=config.kld_weight)

            # kld_tracked.append(kld.detach().cpu())
            # kld_w_tracked.append(kld_weight)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
                
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*config.batch_size))
        wandb.log({"loss": overall_loss / (batch_idx*config.batch_size)})
            
    print("Finished!")

# Hyperparameter opt
sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'goal': 'minimize', 
        'name': 'loss'
        },
    'parameters': {
        'batch_size': {'values': [8, 16, 32, 64]},
        'epochs': {'max': 2000, 'min': 100},
        'learning_rate': {'max': 0.1, 'min': 0.000001},
        'kld_weight': {'max': 0.9, 'min': 0.5},
        'num_layers': {'values': [2, 3, 4]},
        'pool_step': {'values': [2, 4]},
        'kernel_size': {'values': [1, 3, 5, 7, 9]}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="PNS Denoising")

wandb.agent(sweep_id, train)