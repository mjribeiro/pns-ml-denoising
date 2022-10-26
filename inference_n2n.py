from base64 import decode
import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader

# Local imports
from datasets.vagus_dataset import VagusDataset
from models.noise2noise_model import *

# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load vagus dataset
train_dataset = VagusDataset(train=True)
test_dataset  = VagusDataset(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# Define model
input_dim = 2048 # TODO: don't hardcode this, find out from input

print("Setting up coordinate VAE model...")
encoder = Noise2NoiseEncoder()
decoder = Noise2NoiseDecoder(data_length=input_dim)
model = Noise2NoiseModel(encoder=encoder, decoder=decoder)

if torch.cuda.is_available():
    model.cuda()

# Load model weights
PATH = './saved/coordinate_vae.pth'
model.load_state_dict(torch.load(PATH))

# Inference
model.eval()

with torch.no_grad():
    for batch_idx, x in enumerate(test_dataloader):
        x = x.to(device).float()
        model.training = False
        
        x_hat = model(x)

        break # Only run once


# x = x.view(16, 1, input_dim)
plt.plot(x.cpu()[0][0])
plt.plot(x_hat.cpu()[0][0])
plt.show()