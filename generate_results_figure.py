import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import find_peaks

# Local imports
from datasets.vagus_dataset import VagusDatasetN2N

# Function definitions
def rolling_rms(x, N):
    # Source: https://dsp.stackexchange.com/a/74822
    return (pd.DataFrame(abs(x)**2).rolling(N).mean()) ** 0.5 

def calculate_envelope(x):
    peak_idx, _ = find_peaks(x, prominence=0.5)

    # Then return these and plot as line
    return peak_idx



# Load original test data
test_dataset = VagusDatasetN2N(stage="test")

# Load coordinate vae results
cvae_noisy_inputs_ch1 = np.load("./results/cvae_noisy_input_ch1.npy")
cvae_reconstr_ch1 = np.load("./results/cvae_reconstr_ch1.npy")

# Load n2n results
n2n_noisy_targets_ch1 = np.load("./results/n2n_noisy_labels_ch1.npy")
n2n_reconstr_ch1 = np.load("./results/n2n_reconstr_ch1.npy")


# Make the following plots:
# 1) CVAE results overlaid on 50% opacity original
# 2) N2N results overlaid on 50% opacity original
# 3) Blood pressure with envelope
# 4) 20s recordings with moving RMS average similar to Metcalfe 2014 paper

# Indices for zommed in plots
start_factor = 0
end_factor = 15

zoom_start = start_factor
zoom_end = 1024 * end_factor

zoom_cvae_noisy = cvae_noisy_inputs_ch1[zoom_start:zoom_end]
zoom_cvae_reconstr = cvae_reconstr_ch1[zoom_start:zoom_end]

zoom_n2n_noisy = n2n_noisy_targets_ch1[zoom_start:zoom_end]
zoom_n2n_reconstr = n2n_reconstr_ch1[zoom_start:zoom_end]
zoom_time = np.arange(0, len(zoom_n2n_noisy)/100e3, 1/100e3)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)

# 1) CVAE data
ax1.plot(zoom_time, zoom_cvae_noisy, alpha=0.3)
ax1.plot(zoom_time, zoom_cvae_reconstr)

# 2) N2N data
ax2.plot(zoom_time, zoom_n2n_noisy, alpha=0.3)
ax2.plot(zoom_time, zoom_n2n_reconstr)

# 3) BP with envelope
entire_bp_recording = test_dataset.load_bp_data(0, len(test_dataset))

# Use only data that was used for testing (drop_last=True when testing, so not entire dataset was used)
time_full = np.arange(0, len(cvae_reconstr_ch1)/100e3, 1/100e3)
max_idx = len(time_full)

ch1_bp_recording = entire_bp_recording[:, 0, :].flatten()

peaks = calculate_envelope(ch1_bp_recording)
peaks = list(filter(lambda x : x < max_idx, peaks))

peaks_x = [x/100e3 for x in peaks]

ax3.plot(time_full, ch1_bp_recording[:max_idx])
ax3.plot(peaks_x, ch1_bp_recording[peaks], '--')
ax3.set_xlim([-2.5, 22.5])

# 4) Moving RMS window of whole data
N_CVAE = int(1000e-3 * 100e3) # 500 ms in samples
N_N2N = int(1000e-3 * 100e3) # 500 ms in samples

# 4.1) CVAE
cvae_reconstr_ch1_rms = rolling_rms(cvae_reconstr_ch1, N_CVAE) 
ax4.plot(time_full, cvae_reconstr_ch1_rms)
ax4.set_xlim([-2.5, 22.5])

# 4.2) N2N
n2n_reconstr_ch1_rms = rolling_rms(n2n_reconstr_ch1, N_N2N) 
ax5.plot(time_full, n2n_reconstr_ch1_rms)
ax5.set_xlim([-2.5, 22.5])

plt.show()

# TO TRY
# - envelopes for raw data and bandpass filtered
# - VAE with bandpass filtered data (as per original paper)
# - VAE output as Noise2Noise targets

# - Multiple channels as multiple Noise2Noise pairs (different noise on each channel, maybe use adjacent pairs but look at amplitudes in spontaneous data first, adjust for time difference)