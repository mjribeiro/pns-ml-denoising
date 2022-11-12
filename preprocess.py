import mat73
import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Local imports
from utils.preprocessing import *


# ----- DATA LOADING
# TODO: Add automated/scripted way of copying this over
file_path = './data/Metcalfe-2014/baselineFast.mat'

recording = mat73.loadmat(file_path)
recording = recording['rawdata']
recording_df = pd.DataFrame(recording, columns=["Time", "Channel 1", "Channel 2", "Channel 3", 
                                                "Channel 4", "Channel 5", "Channel 6", "Channel 7",
                                                "Channel 8", "Channel 9"])

# Visualise first few rows of data
# print(recording_df.head())

# ----- PLOT SAMPLE OF ALL CHANNELS (STACKED)
# plot_all_channels(recording_df, filt=False, fs=100e3, lims=np.asarray([0, 2]))

# ----- SORT ALL CHANNEL DATA INTO ONE DATASET
vagus_data_raw, vagus_data_bp_wide, vagus_data_bp_narrow = generate_datasets(recording_df, 
                                                                             fs=100e3, 
                                                                             num_channels=9, 
                                                                             win_length=0.008,
                                                                             flattened=False)
# plt.plot(vagus_data_raw[10, 0, :])
# plt.plot(vagus_data_bp_wide[10, 0, :])
# plt.plot(vagus_data_bp_narrow[10, 0, :])
# plt.show()

# ----- SAVE DATASET TO BE USED IN ML
vagus_train_raw, vagus_test_raw             = train_test_split(vagus_data_raw, test_size=0.2, random_state=1)
vagus_train_bp_wide, vagus_test_bp_wide     = train_test_split(vagus_data_bp_wide, test_size=0.2, random_state=1)
vagus_train_bp_narrow, vagus_test_bp_narrow = train_test_split(vagus_data_bp_narrow, test_size=0.2, random_state=1)

# Save as numpy arrays?
np.save('./data/Metcalfe-2014/vagus_train_raw.npy', vagus_train_raw)
np.save('./data/Metcalfe-2014/vagus_test_raw.npy', vagus_test_raw)

np.save('./data/Metcalfe-2014/vagus_train_bp_wide.npy', vagus_train_bp_wide)
np.save('./data/Metcalfe-2014/vagus_test_bp_wide.npy', vagus_test_bp_wide)

np.save('./data/Metcalfe-2014/vagus_train_bp_narrow.npy', vagus_train_bp_narrow)
np.save('./data/Metcalfe-2014/vagus_test_bp_narrow.npy', vagus_test_bp_narrow)