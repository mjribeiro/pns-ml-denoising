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
from utils.analysis import *



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
                "learning_rate": 0.0001,
                "epochs": 1000,
                "batch_size": 1024,
                "kernel_size": 3,
                "change_weights": False})

    # Load vagus dataset
    train_dataset = VagusDatasetN2N(stage="train")
    val_dataset   = VagusDatasetN2N(stage="val")
    test_dataset  = VagusDatasetN2N(stage="test")

    wandb.init(project="PNS Denoising")
    config = wandb.config

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_dataloader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataloader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)

    sample, _, _ = train_dataset[0]

    # Define model
    print("Setting up Noise2Noise model...")
    encoder = Noise2NoiseEncoder(num_channels=sample.shape[0])
    decoder = Noise2NoiseDecoder(num_channels=sample.shape[0])
    model   = Noise2NoiseModel(encoder=encoder, decoder=decoder).to(device)

    # Hyperparameter setup
    mse_loss = nn.MSELoss()

    def loss_function(x, x_hat, bp, device, epoch, idx, change_weights=False):

        MSE = mse_loss(x, x_hat)

        # Adapt loss based on current epoch
        if change_weights:
            loss_step = config.epochs - (config.epochs / 4)
            loss_weight = np.linspace(0.01, 0.75, int(loss_step))

            if epoch < (config.epochs / 4):
                loss = MSE
            else:
                bp_envelope, x_hat_moving_rms = get_rms_envelope(x_hat=x_hat, bp=bp, window_len=int(100e3), device=device)

                MSE_envelope = mse_loss(bp_envelope, x_hat_moving_rms)

                MSE_envelope_weight = loss_weight[idx]
                MSE_reconstr_weight = 1 - MSE_envelope_weight

                loss =  (MSE_envelope_weight * MSE_envelope) + (MSE_reconstr_weight * MSE)
        else:
            # Get moving RMS plots
            bp_envelope, x_hat_moving_rms = get_rms_envelope(x_hat=x_hat, bp=bp, window_len=int(100e3), device=device)
            MSE_envelope = mse_loss(bp_envelope, x_hat_moving_rms)

            MSE_envelope_weight = 0.25
            MSE_reconstr_weight = 0.75

            loss =  (MSE_envelope_weight * MSE_envelope) + (MSE_reconstr_weight * MSE)
        
        return loss


    optimizer = Adam(model.parameters(),
                     lr=config.learning_rate)
    wandb.watch(model, log="all")

    # Training
    print("Start training Noise2Noise model...")

    if torch.cuda.is_available():
        model.cuda()

    # Get parameter count
    print(sum(p.numel() for p in model.parameters()))

    # ----- TRAINING/VALIDATION LOOP -----
    training_start_time = time.time()

    # best_loss = 99999.0
    # best_loss_epoch = 0

    idx = 0

    for epoch in range(config.epochs):
        model.train()
        overall_loss = 0

        # ----- TRAIN -----
        for _, (inputs, targets, bp) in enumerate(train_dataloader):
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            optimizer.zero_grad()

            x_hat = model(inputs)
            loss = loss_function(targets, x_hat, bp, device=device, epoch=epoch, idx=idx, change_weights=config.change_weights)

            overall_loss += loss.item() * inputs.size(0)

            loss.backward()
            optimizer.step()

        # average_loss = overall_loss / (batch_idx*config.batch_size)
        # Sources for new loss: https://stackoverflow.com/questions/61284248/is-it-a-good-idea-to-multiply-loss-item-by-batch-size-to-get-the-loss-of-a-bat
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        average_loss = overall_loss / len(train_dataset)

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", average_loss)
        wandb.log({"loss": average_loss})
    
        # ----- VALIDATE -----
        model.eval()

        with torch.no_grad():
            overall_val_loss = 0
            for _, (inputs, targets, bp) in enumerate(val_dataloader):
                inputs = inputs.to(device).float()
                targets = targets.to(device).float()

                x_hat = model(inputs)
                loss = loss_function(targets, x_hat, bp, device=device, epoch=epoch, idx=idx, change_weights=config.change_weights)

                overall_val_loss += loss.item() * inputs.size(0)

            average_val_loss = overall_val_loss / len(val_dataset)
            print("\t\t\t\tAverage Validation Loss: ", average_val_loss)
            wandb.log({"val_loss": average_val_loss})

        # Index that updates with each epoch past a certain point
        # Needed for updating weights for loss elements
        if epoch >= (config.epochs / 4): 
            idx += 1

    print("Finished!")
    print('Training finished, took {:.2f}s'.format(time.time() - training_start_time))

    Path('./saved/').mkdir(parents=True, exist_ok=True)
    # torch.save(best_model.state_dict(), './saved/noise2noise.pth')

    
    # ----- INFERENCE -----
    with torch.no_grad():
        x_hats = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
        xs = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))
        xs_cleaner = np.zeros((len(test_dataloader)*config.batch_size, 9, 1024))

        start_idx = 0
        end_idx = start_idx+config.batch_size

        for batch_idx, (inputs, targets, _) in enumerate(test_dataloader):
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

    for i in range(9):
        np.save(f"./results/n2n_noisy_input_ch{i+1}.npy", xs[:, i, :].flatten())
        np.save(f"./results/n2n_noisy_labels_ch{i+1}.npy", xs_cleaner[:, i, :].flatten())
        np.save(f"./results/n2n_reconstr_ch{i+1}.npy", x_hats[:, i, :].flatten())

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
