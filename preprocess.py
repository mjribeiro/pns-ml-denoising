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
print("Is file in path? ", os.path.isfile(file_path)) # Check if docker mounted volume is working

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
vagus_dataset = generate_dataset(recording_df, fs=100e3, num_channels=9, win_length=0.02048)
# plt.plot(vagus_dataset[0])
# plt.show()

# ----- SAVE DATASET TO BE USED IN ML
vagus_train, vagus_test = train_test_split(vagus_dataset, test_size=0.2, random_state=1)
np.save('./data/Metcalfe-2014/vagus_train.npy', vagus_train)
np.save('./data/Metcalfe-2014/vagus_test.npy', vagus_test)