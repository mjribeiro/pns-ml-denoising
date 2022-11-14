import matplotlib.pyplot as plt
import numpy as np
import wandb

from torch.utils.data import DataLoader

# Local imports
from datasets.vagus_dataset import VagusDatasetN2N
from models.noise2noise_model import *

# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load vagus dataset
train_dataset = VagusDatasetN2N(train=True)
test_dataset  = VagusDatasetN2N(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

sample, target = train_dataset.__getitem__(0)

# # Plot single channel
# plt.plot(sample[0, :])
# plt.plot(target[0, :])
# plt.show()

# Define model
print("Setting up Noise2Noise model...")
encoder = Noise2NoiseEncoder(num_channels=sample.shape[0])
decoder = Noise2NoiseDecoder(num_channels=sample.shape[0], data_length=len(sample[0, :]))
model = Noise2NoiseModel(encoder=encoder, decoder=decoder).to(device)

if torch.cuda.is_available():
    model.cuda()

# Load model weights
PATH = './saved/noise2noise.pth'
model.load_state_dict(torch.load(PATH))

# Inference
model.eval()

with torch.no_grad():
    x_hats = np.zeros((len(test_dataloader), 9, 1024))
    xs = np.zeros((len(test_dataloader), 9, 1024))
    xs_cleaner = np.zeros((len(test_dataloader), 9, 1024))
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        model.training = False
        
        x_hat = model(inputs)
        x_hat = x_hat.cpu().numpy()

        x_hats[batch_idx, :, :] = x_hat[0]
        xs[batch_idx, :, :] = inputs[0].cpu().numpy()
        xs_cleaner[batch_idx, :, :] = targets[0].cpu().numpy()

# x = x.view(16, 1, input_dim)
for i in range(9):
    plt.plot(xs[:, i, :].flatten(), label=f"Noisy input {i}")
    plt.plot(xs_cleaner[:, i, :].flatten(), label=f"Noisy labels {i}")
    plt.plot(x_hats[:, i, :].flatten(), label=f"Reconstructed {i}")
plt.show()