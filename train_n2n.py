import copy
import gc
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import wandb
import random

from pathlib import Path
from torch.optim import Adam
from torch.utils.data import DataLoader

# Local imports
from models.noise2noise_model import *
from datasets.vagus_dataset import VagusDatasetN2N



def train():
    matplotlib.use('agg')

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Address GPU memory issues (source: https://stackoverflow.com/a/66921450)
    gc.collect()
    torch.cuda.empty_cache()

    # Select GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Weights&Biases initialisation
    wandb.init(project="PNS Denoising",
            config = {
                "learning_rate": 0.00001,
                "epochs": 170,
                "batch_size": 1024,
                "kernel_size": 3})

    # Load vagus dataset
    train_dataset = VagusDatasetN2N(stage="train")
    val_dataset   = VagusDatasetN2N(stage="val")
    test_dataset  = VagusDatasetN2N(stage="test")

    wandb.init(project="PNS Denoising")
    config = wandb.config

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    sample, _ = train_dataset[0]

    # Define model
    print("Setting up Noise2Noise model...")
    encoder = Noise2NoiseEncoder(num_channels=sample.shape[0])
    decoder = Noise2NoiseDecoder(num_channels=sample.shape[0])
    model   = Noise2NoiseModel(encoder=encoder, decoder=decoder).to(device)

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

    # Get parameter count
    # print(sum(p.numel() for p in model.parameters()))

    # ----- TRAINING -----
    training_start_time = time.time()
    model.train()

    best_loss = 99999.0
    best_loss_epoch = 0

    for epoch in range(config.epochs):
        overall_loss = 0

        for _, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()

            x_hat = model(inputs)
            loss = loss_function(targets, x_hat)

            overall_loss += loss.item() * inputs.size(0)

            loss.backward()
            optimizer.step()

        # average_loss = overall_loss / (batch_idx*config.batch_size)
        # Sources for new loss: https://stackoverflow.com/questions/61284248/is-it-a-good-idea-to-multiply-loss-item-by-batch-size-to-get-the-loss-of-a-bat
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        average_loss = overall_loss / len(train_dataset)

        if average_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = average_loss
            best_loss_epoch = epoch

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", average_loss)
        wandb.log({"loss": average_loss})

        if epoch > best_loss_epoch + 20:
            break

    print("Finished!")
    print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))

    Path('./saved/').mkdir(parents=True, exist_ok=True)
    # torch.save(best_model.state_dict(), './saved/noise2noise.pth')

    
    # ----- VALIDATION FOR HYPERPARAMETER OPT -----
    # model.load_state_dict(torch.load('./saved/noise2noise.pth'))
    model = best_model

    model.eval()

    with torch.no_grad():
        overall_val_loss = 0
        for _, (inputs, targets) in enumerate(val_dataloader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            x_hat = model(inputs)
            loss = loss_function(targets, x_hat)

            overall_val_loss += loss.item() * inputs.size(0)

        average_val_loss = overall_val_loss / len(val_dataset)
        print("\tAverage Validation Loss: ", average_val_loss)
        wandb.log({"val_loss": average_val_loss})

    
    # ----- INFERENCE -----
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

    time_plot = np.arange(0, len(xs[1000:1004, 0, :].flatten())/100e3, 1/100e3)

    start_plot_sample = 1000
    end_plot_sample = 1004
    xs_y = xs[start_plot_sample:end_plot_sample, 0, :].flatten()
    xs_cleaner_y =xs_cleaner[start_plot_sample:end_plot_sample, 0, :].flatten()
    x_hats_y = x_hats[start_plot_sample:end_plot_sample, 0, :].flatten()
    # ys = [xs_y, xs_cleaner_y, x_hats_y]

    fig, ax = plt.subplots()
    ax.plot(time_plot, xs_y, label=f"Noisy input")
    ax.plot(time_plot, xs_cleaner_y, label=f"Noisy labels")
    ax.plot(time_plot, x_hats_y, label=f"Reconstructed")
    ax.legend()
    # fig.show()

    wandb.log({"plot": fig})

    np.save("./results/n2n_noisy_input_ch1.npy", xs[:, 0, :].flatten())
    np.save("./results/n2n_noisy_labels_ch1.npy", xs_cleaner[:, 0, :].flatten())
    np.save("./results/n2n_reconstr_ch1.npy", x_hats[:, 0, :].flatten())

    wandb.finish()

train()

# # ----- HYPERPARAMETER OPT -----
# sweep_configuration = {
#     'method': 'bayes',
#     'metric': {
#         'goal': 'minimize',
#         'name': 'val_loss'
#         },
#     'parameters': {
#         'batch_size': {'values': [1024]},
#         'epochs': {'max': 500, 'min': 10},
#         'learning_rate': {'distribution': 'inv_log_uniform_values', 'max': 0.1, 'min': 0.000001},
#         'kernel_size': {'values': [1, 3, 5, 7, 9]}
#     }
# }

# sweep_id = wandb.sweep(sweep=sweep_configuration, project="PNS Denoising")

# wandb.agent(sweep_id, train)
