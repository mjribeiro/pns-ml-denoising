import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import make_interp_spline
from scipy.signal import butter, filtfilt, find_peaks

# Local imports
from utils.preprocessing import *
from datasets.vagus_dataset import VagusDatasetN2N

# Function definitions
def rolling_rms(x, N):
    # Source: https://dsp.stackexchange.com/a/74822
    return (pd.DataFrame(abs(x)**2).rolling(N).mean()) ** 0.5 

def calculate_envelope_points(x):
    peak_idx, _ = find_peaks(x, prominence=0.5)

    # Then return these and plot as line
    return peak_idx

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Load original test data
test_dataset = VagusDatasetN2N(stage="test")

# Load coordinate vae results
cvae_noisy_inputs_ch1 = np.load("./results/cvae_noisy_input_ch1.npy")
cvae_reconstr_ch1 = np.load("./results/cvae_reconstr_ch1.npy")

# Load n2n results
n2n_noisy_targets_ch1 = np.load("./results/n2n_noisy_labels_ch1.npy")
n2n_reconstr_ch1 = np.load("./results/n2n_reconstr_ch1.npy")


# Make the following plots:
# 1) Original data and results with bandpass filtering
# 2) CVAE results overlaid on 50% opacity original
# 3) N2N results overlaid on 50% opacity original

# On a new plot:
# 1) Blood pressure with envelope
# 2) 20s recordings with moving RMS average similar to Metcalfe 2014 paper

fig1, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig1.tight_layout()

fig2, (ax4, ax5, ax6, ax7) = plt.subplots(4, 1)
fig1.tight_layout()

# Indices for zommed in plots
start_factor = 0
end_factor = 15

zoom_start = start_factor
zoom_end = 1024 * end_factor

zoom_cvae_noisy = cvae_noisy_inputs_ch1[zoom_start:zoom_end]
zoom_time = np.arange(0, len(zoom_cvae_noisy)/100e3, 1/100e3)

# 1) Original plus bandpass filtered
orig_filtered_win = bandpass_filter(zoom_cvae_noisy, [100, 5000], 100e3)
orig_filtered = bandpass_filter(cvae_noisy_inputs_ch1, [100, 5000], 100e3)

ax1.plot(zoom_time, zoom_cvae_noisy, alpha=0.3)
ax1.plot(zoom_time, orig_filtered_win)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude (norm.)")

zoom_cvae_reconstr = cvae_reconstr_ch1[zoom_start:zoom_end]

zoom_n2n_noisy = n2n_noisy_targets_ch1[zoom_start:zoom_end]
zoom_n2n_reconstr = n2n_reconstr_ch1[zoom_start:zoom_end]

# 2) CVAE data
ax2.plot(zoom_time, zoom_cvae_noisy, alpha=0.3)
ax2.plot(zoom_time, zoom_cvae_reconstr)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude (norm.)")

# 3) N2N data
ax3.plot(zoom_time, zoom_n2n_noisy, alpha=0.3)
ax3.plot(zoom_time, zoom_n2n_reconstr)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Amplitude (norm.)")

# 4) BP with envelope
entire_bp_recording = test_dataset.load_bp_data(0, len(test_dataset))

# Use only data that was used for testing (drop_last=True when testing, so not entire dataset was used)
time_full = np.arange(0, len(cvae_reconstr_ch1)/100e3, 1/100e3)
max_idx = len(time_full)

ch1_bp_recording = entire_bp_recording[:, 0, :].flatten()


peaks = calculate_envelope_points(ch1_bp_recording)
peaks = list(filter(lambda x : x < max_idx, peaks))

peaks_x = [x/100e3 for x in peaks]

# Make spline from envelope points
spl = make_interp_spline(peaks_x, ch1_bp_recording[peaks])
xnew = np.linspace(min(peaks_x), max(peaks_x), 1000)
spl_y = spl(xnew)

# Lowpass filter envelope so it's smoother
bp_envelope = butter_lowpass_filter(spl_y, 15, 1000)

ax4.plot(time_full, ch1_bp_recording[:max_idx])
ax4.plot(xnew, bp_envelope)
ax4.set_xlim([-2.5, 22.5])
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Amplitude")

# 5) Moving RMS window of bandpass filtered data
N_BANDPASS = int(1000e-3 * 100e3) # 1000 ms in samples
bandpass_rms = rolling_rms(orig_filtered, N_BANDPASS)
ax5.plot(time_full, bandpass_rms)
ax5.set_xlim([-2.5, 22.5])
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Amplitude (norm.)")

# 6) Moving RMS window of whole data
N_CVAE = int(1000e-3 * 100e3) # 1000 ms in samples
N_N2N = int(1000e-3 * 100e3) # 1000 ms in samples

# 6) CVAE
cvae_reconstr_ch1_rms = rolling_rms(cvae_reconstr_ch1, N_CVAE) 
ax6.plot(time_full, cvae_reconstr_ch1_rms)
ax6.set_xlim([-2.5, 22.5])
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Amplitude (norm.)")

# 7) N2N
n2n_reconstr_ch1_rms = rolling_rms(n2n_reconstr_ch1, N_N2N) 
ax7.plot(time_full, n2n_reconstr_ch1_rms)
ax7.set_xlim([-2.5, 22.5])
ax7.set_xlabel("Time (s)")
ax7.set_ylabel("Amplitude (norm.)")

plt.show()

# TO TRY
# - envelopes for raw data and bandpass filtered
# - VAE with bandpass filtered data (as per original paper)
# - VAE output as Noise2Noise targets

# - Multiple channels as multiple Noise2Noise pairs (different noise on each channel, maybe use adjacent pairs but look at amplitudes in spontaneous data first, adjust for time difference)