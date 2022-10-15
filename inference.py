import matplotlib.pyplot as plt
import wandb

from torch.utils.data import DataLoader

# Local imports
from datasets.vagus_dataset import VagusDataset
from models.vae_model import *

# Select GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load vagus dataset
train_dataset = VagusDataset(train=True)
test_dataset  = VagusDataset(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Define model
input_dim = 2048 # TODO: don't hardcode this, find out from input
hidden_dim_encoder = 256
hidden_dim_decoder = 256
kernel_size = 1

print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=1, latent_dim=1, kernel_size=kernel_size, device=device)
decoder = Decoder(latent_dim=1, output_dim=1, kernel_size=kernel_size, device=device)
model = CoordinateVAEModel(Encoder=encoder, Decoder=decoder, device=device)

# Load model weights
PATH = './saved/coordinate_vae.pth'
model.load_state_dict(torch.load(PATH))

# Inference
model.eval()

with torch.no_grad():
    for batch_idx, x in enumerate(test_dataloader):
        # x = x.view(batch_size, x_dim)
        x = x.to(device).float()
        
        x_hat = model(x)

        break # Only run once


# x = x.view(16, 1, input_dim)
plt.plot(x[0][0])
plt.plot(x_hat[0][0])
plt.show()