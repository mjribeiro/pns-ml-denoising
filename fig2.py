import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from array import array

# Local imports
from datasets.vagus_dataset import VagusDataset

# Load vagus ENG data
train_dataset = VagusDataset(train=True)
test_dataset  = VagusDataset(train=False)

# Load arterial blood pressure data
# TODO: Ask Ben about gains and whether any filtering was done
ecg_record = wfdb.rdrecord('data/Metcalfe-2014/ECG', sampfrom=0, sampto=20*(100000))
ecg_signal = ecg_record.p_signal

plt.plot(ecg_signal)
plt.show()

# TODO: Bandpass filter vagus ENG

# TODO: Load trained models for coordinate VAE and noise2noise

# TODO: Run inference on same input for both

# TODO: Plot bandpass filtered data, coord vae reconstruction, noise2noise reconstr, arterial blood pressure (single column of plots)