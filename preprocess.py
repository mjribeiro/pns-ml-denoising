import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.io

from pathlib import Path

# Local imports
from utils.noise_sources.gaussian_white_noise import GaussianWhiteNoise
from utils.noise_sources.brownian_noise import BrownianNoise

# ----- DATA LOADING
file_path = './data/Metcalfe-2020/Resting/1551.mat'
print("Is file in path? ", os.path.isfile(file_path)) # Check if docker mounted volume is working

recording = scipy.io.loadmat(file_path)
recording = recording['rawdata']
recording_df = pd.DataFrame(recording, columns=["Time", "Channel 1", "Channel 2", "Channel 3", 
                                               "Channel 4", "Channel 5"])

# Visualise first few rows of data
recording_df.head()

# ----- ADDING NOISE
# # Apply varying kinds of noise to neural recordings (at the moment, a combination of White Gaussian and Brownian)
gauss_standard_dev_range = np.linspace(0.08, 0.1, 1)

mean = 0
time = recording_df["Time"]

input_signals = np.zeros((len(recording_df.columns)-1, len(time)))
target_signals = recording_df.drop(['Time'], axis=1).to_numpy().T # from original, high SNR recordings

for channel, index in zip(recording_df.columns[1:], range(0,len(recording_df.columns[1:]))):
    original = recording_df[channel]
    
    # Generate and appy noise - GWN
    standard_dev = np.random.choice(gauss_standard_dev_range)
    noise_profile_GWN = GaussianWhiteNoise(len(original), standard_dev, mean)
    generated_GWN = noise_profile_GWN.generate()
    input_signal_GWN = noise_profile_GWN.apply(original)

    # Generate and apply noise - BN
    noise_profile_BN = BrownianNoise(len(input_signal_GWN)) 
    generated_BN = noise_profile_BN.generate()
    input_signal = noise_profile_BN.apply(input_signal_GWN)
    input_signals[index] = input_signal
    
# ----- PLOT SINGLE RECORDING TO VERIFY
plt.plot(input_signals[0])
plt.plot(target_signals[0, :])
plt.show()

# ----- SAVE NOISY DATA
save_path = './data/Metcalfe-2020/saved/'
Path(save_path).mkdir(parents=True, exist_ok=True)
np.save(save_path + 'input_signals.npy', input_signals)
np.save(save_path + 'target_signals.npy', target_signals)