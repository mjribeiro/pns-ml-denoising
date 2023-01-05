from torch.utils.data import DataLoader
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Temp
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Local imports
from models.cvae_model import *
from datasets.vagus_dataset import VagusDataset

# Load vagus dataset
train_dataset = VagusDataset(train=True)
test_dataset  = VagusDataset(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True)

sample = train_dataset.__getitem__(5)

fig, ax = plt.subplots(9)
colors = plt.cm.Set1.colors

for idx, color in zip(range(np.shape(sample)[0]), colors):
    ax[idx].plot(sample[idx], color=color)
    ax[idx].axis('off')

ax[idx].axis('on')
plt.show()
