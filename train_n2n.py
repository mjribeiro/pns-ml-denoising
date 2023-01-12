import copy
import gc
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb

from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader

# Local imports
from models.noise2noise_model import *
from datasets.vagus_dataset import VagusDatasetN2N



def train():
    # Address GPU memory issues (source: https://stackoverflow.com/a/66921450)
    gc.collect()
    torch.cuda.empty_cache()

    # Select GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Weights&Biases initialisation
    # wandb.init(project="PNS Denoising",
    #         config = {
    #             "learning_rate": 0.0001,
    #             "epochs": 1,
    #             "batch_size": 64,
    #             "kernel_size": 9})
    # config = wandb.config
    # wandb.init(project="PNS Denoising")

    # Load vagus dataset
    train_dataset = VagusDatasetN2N(train=True)
    test_dataset  = VagusDatasetN2N(train=False)

    wandb.init(project="PNS Denoising")
    config = wandb.config

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    sample, _ = train_dataset[0]

    # Define model
    print("Setting up Noise2Noise model...")
    encoder = Noise2NoiseEncoder(num_channels=sample.shape[0])
    decoder = Noise2NoiseDecoder(num_channels=sample.shape[0])
    model = Noise2NoiseModel(encoder=encoder, decoder=decoder).to(device)

    # Hyperparameter setup
    mse_loss = nn.MSELoss()

    def loss_function(x, x_hat):
        return mse_loss(x, x_hat)


    optimizer = Adam(model.parameters(), 
                    lr=config.learning_rate)
    wandb.watch(model, log="all")

    # Training
    print("Start training Noise2Noise model...")

    if torch.cuda.is_available():
        model.cuda()

    # ----- TRAINING -----
    model.train()

    best_loss = 99999.0

    for epoch in range(config.epochs):
        overall_loss = 0

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()

            x_hat = model(inputs)
            loss = loss_function(targets, x_hat)

            if loss < best_loss:
                best_model = copy.deepcopy(model)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*config.batch_size))
        wandb.log({"loss": overall_loss / (batch_idx*config.batch_size)})
            
    print("Finished!")

    Path('./saved/').mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), './saved/noise2noise.pth')

    # ----- INFERENCE -----
    # model.load_state_dict(torch.load('./saved/noise2noise.pth'))
    model = best_model

    # Inference
    model.eval()

    with torch.no_grad():
        x_hats = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
        xs = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
        xs_cleaner = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))

        start_idx = 0
        end_idx = start_idx+config.batch_size

        for batch_idx, (inputs, targets) in enumerate(test_dataloader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            x_hat = model(inputs)

            x_hats[start_idx:end_idx, :, :] = x_hat.cpu().numpy()
            xs[start_idx:end_idx, :, :] = inputs.cpu().numpy()
            xs_cleaner[start_idx:end_idx, :, :] = targets.cpu().numpy()

            start_idx += config.batch_size
            end_idx += config.batch_size

    time = np.arange(0, len(xs[1000:1004, 0, :].flatten())/100e3, 1/100e3)
    # plt.plot(time, xs[1000:1004, 0, :].flatten(), label=f"Noisy input")
    # plt.plot(time, xs_cleaner[1000:1004, 0, :].flatten(), label=f"Noisy labels")
    # plt.plot(time, x_hats[1000:1004, 0, :].flatten(), label=f"Reconstructed")
    # plt.legend()
    # plt.savefig(f'./saved/n2n_{sweep_id}.png')
    xs_y = xs[1000:1004, 0, :].flatten()
    xs_cleaner_y =xs_cleaner[1000:1004, 0, :].flatten()
    x_hats_y = x_hats[1000:1004, 0, :].flatten()
    ys = [xs_y, xs_cleaner_y, x_hats_y]
    
    # data = [[x, y] for (x, y) in zip(time, ys)]
    # table = wandb.Table(data=data, columns = ["x", "y"])
    wandb.log({"my_custom_plot_id" : wandb.plot.line_series(xs=time, ys=ys,
               keys=["Noisy input","Noisy labels","Reconstructed"], title="Original signals vs reconstruction", xname="Time (s)")})

    # wandb.finish()
# sweep_id = 1289
# train()

# ----- HYPERPARAMETER OPT -----
sweep_configuration = {
    'method': 'bayes',
    'metric': {
        'goal': 'minimize', 
        'name': 'loss'
        },
    'parameters': {
        'batch_size': {'values': [8, 16, 32, 64]},
        'epochs': {'max': 500, 'min': 10},
        'learning_rate': {'max': 0.1, 'min': 0.000001},
        'kernel_size': {'values': [1, 3, 5, 7, 9]}
    }
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="PNS Denoising")

wandb.agent(sweep_id, train)