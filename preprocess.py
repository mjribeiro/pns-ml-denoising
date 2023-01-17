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

print("Loading recordings...")
recording      = mat73.loadmat(file_path)
eng_data       = recording['rawdata']
blood_pressure = recording['bloodPressure']
eng_data_df    = pd.DataFrame(eng_data, columns=["Time", "Channel 1", "Channel 2", "Channel 3", 
                                                "Channel 4", "Channel 5", "Channel 6", "Channel 7",
                                                "Channel 8", "Channel 9"])

# Visualise first few rows of data
# print(recording_df.head())

# ----- PLOT SAMPLE OF ALL CHANNELS (STACKED)
# plot_all_channels(eng_data_df, filt=False, fs=100e3, lims=np.asarray([0, 2]))
win_length = 0.01024
fs = 100e3

# ----- SORT ALL CHANNEL DATA INTO ONE DATASET
print("Creating datasets...")
vagus_data_raw, vagus_data_filt_wide, vagus_data_filt_narrow = generate_datasets(eng_data_df,
                                                                                 bp_data=blood_pressure,
                                                                                 fs=fs, 
                                                                                 num_channels=9, 
                                                                                 win_length=win_length)

# TODO: Make test set hold out - maybe split before prepare_n2n_data and only do this for training data?
n2n_X_train, n2n_X_val, n2n_X_test, n2n_y_train, n2n_y_val, n2n_y_test = prepare_n2n_data(noisy_inputs=vagus_data_raw, 
                                                                                          filtered_data=vagus_data_filt_narrow, 
                                                                                          bp_data=blood_pressure)

# ----- SAVE DATASET TO BE USED IN ML
np.save(f'./data/Metcalfe-2014/n2n_X_train.npy', n2n_X_train)
np.save(f'./data/Metcalfe-2014/n2n_X_val.npy', n2n_X_val)
np.save(f'./data/Metcalfe-2014/n2n_X_test.npy', n2n_X_test)
np.save(f'./data/Metcalfe-2014/n2n_y_train.npy', n2n_y_train)
np.save(f'./data/Metcalfe-2014/n2n_y_val.npy', n2n_y_val)
np.save(f'./data/Metcalfe-2014/n2n_y_test.npy', n2n_y_test)

# Older train/test splits, not currently being used
# vagus_train_raw, vagus_test_raw                 = train_test_split(vagus_data_raw, test_size=0.2, shuffle=False)
# vagus_train_filt_wide, vagus_test_filt_wide     = train_test_split(vagus_data_filt_wide, test_size=0.2, shuffle=False)
# vagus_train_filt_narrow, vagus_test_filt_narrow = train_test_split(vagus_data_filt_narrow, test_size=0.2, shuffle=False)

# print("Saving datasets...")
# # Save as numpy arrays?
# np.save(f'./data/Metcalfe-2014/vagus_train_raw_{int(win_length*fs)}.npy', vagus_train_raw)
# np.save(f'./data/Metcalfe-2014/vagus_test_raw_{int(win_length*fs)}.npy', vagus_test_raw)

# np.save(f'./data/Metcalfe-2014/vagus_train_filt_wide_{int(win_length*fs)}.npy', vagus_train_filt_wide)
# np.save(f'./data/Metcalfe-2014/vagus_test_filt_wide_{int(win_length*fs)}.npy', vagus_test_filt_wide)

# np.save(f'./data/Metcalfe-2014/vagus_train_filt_narrow_{int(win_length*fs)}.npy', vagus_train_filt_narrow)
# np.save(f'./data/Metcalfe-2014/vagus_test_filt_narrow_{int(win_length*fs)}.npy', vagus_test_filt_narrow)
