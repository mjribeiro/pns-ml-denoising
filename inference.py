import matplotlib.pyplot as plt
import numpy as np
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

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Define model
input_chs = 9 # TODO: don't hardcode this, find out from input
kernel_size = 3

print("Setting up coordinate VAE model...")
encoder = Encoder(input_dim=input_chs, 
                latent_dim=100, 
                kernel_size=3, 
                num_layers=4, 
                pool_step=2, 
                batch_size=32, 
                device=device)
decoder = Decoder(latent_dim=100, 
                output_dim=input_chs, 
                kernel_size=3, 
                num_layers=4, 
                pool_step=2, 
                device=device)
model = CoordinateVAEModel(Encoder=encoder, 
                        Decoder=decoder)

if torch.cuda.is_available():
    model.cuda()

# Load model weights
PATH = './saved/coordinate_vae.pth'
model.load_state_dict(torch.load(PATH))

# Inference
model.eval()

with torch.no_grad():
    x_hats = np.zeros((len(test_dataloader), 9, 800))
    xs = np.zeros((len(test_dataloader), 9, 800))
    for batch_idx, x in enumerate(test_dataloader):
        x = x.to(device).float()
        model.training = False
        
        x_hat = model(x)
        x_hat = x_hat.cpu().numpy()

        x_hats[batch_idx, :, :] = x_hat[0]
        xs[batch_idx, :, :] = x[0].cpu().numpy()

# x = x.view(16, 1, input_dim)
plt.plot(xs[:, 0, :].flatten())
plt.plot(x_hats[:, 0, :].flatten())
plt.show()