import matplotlib.pyplot as plt
import mat73
import numpy as np
import os
import pandas as pd
import scipy.io

from pathlib import Path
from scipy import signal

# Local imports
from utils.noise_sources.gaussian_white_noise import GaussianWhiteNoise
from utils.noise_sources.brownian_noise import BrownianNoise
from utils.preprocessing import *

# ----- DATA LOADING
# TODO: Add automated/scripted way of copying this over
file_path = './data/Metcalfe-2014/baselineFast.mat'
print("Is file in path? ", os.path.isfile(file_path)) # Check if docker mounted volume is working

recording = mat73.loadmat(file_path)
recording = recording['rawdata']
recording_df = pd.DataFrame(recording, columns=["Time", "Channel 1", "Channel 2", "Channel 3", 
                                                "Channel 4", "Channel 5", "Channel 6", "Channel 7",
                                                "Channel 8", "Channel 9"])

# Visualise first few rows of data
# print(recording_df.head())

# Filtering
y = recording_df["Channel 1"]

y = remove_artefacts(y)
y_filt = lowpass_filter(y, lowcut=10e3, fs=100e3, order=4)

# ----- PLOT ORIGINAL RECORDING FROM ONE CHANNEL
gain = 10 ** (94/20) # 94 dB, so 10^(gain_dB / 20) for voltage ratio?
plt.plot(recording_df["Time"], y_filt / gain)
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("Low-pass filtered and artefact-free recording (ch1)")
plt.show()

# ----- TODO NEXT: PLOT ALL CHANNELS ON TOP OF EACH OTHER

# ----- ADDING NOISE
# # Apply varying kinds of noise to neural recordings (at the moment, a combination of White Gaussian and Brownian)
# gauss_standard_dev_range = np.linspace(0.08, 0.1, 1)

# mean = 0
# time = recording_df["Time"]

# input_signals = np.zeros((len(recording_df.columns)-1, len(time)))
# target_signals = recording_df.drop(['Time'], axis=1).to_numpy().T # from original, high SNR recordings

# for channel, index in zip(recording_df.columns[1:], range(0,len(recording_df.columns[1:]))):
#     original = recording_df[channel]
    
#     # Generate and appy noise - GWN
#     standard_dev = np.random.choice(gauss_standard_dev_range)
#     noise_profile_GWN = GaussianWhiteNoise(len(original), standard_dev, mean)
#     generated_GWN = noise_profile_GWN.generate()
#     input_signal_GWN = noise_profile_GWN.apply(original)

#     # Generate and apply noise - BN
#     noise_profile_BN = BrownianNoise(len(input_signal_GWN)) 
#     generated_BN = noise_profile_BN.generate()
#     input_signal = noise_profile_BN.apply(input_signal_GWN)
#     input_signals[index] = input_signal
    
# # ----- PLOT SINGLE RECORDING TO VERIFY
# plt.plot(input_signals[0])
# plt.plot(target_signals[0, :])
# plt.show()

# # ----- SAVE NOISY DATA
# save_path = './data/Metcalfe-2020/saved/'
# Path(save_path).mkdir(parents=True, exist_ok=True)
# np.save(save_path + 'input_signals.npy', input_signals)
# np.save(save_path + 'target_signals.npy', target_signals)